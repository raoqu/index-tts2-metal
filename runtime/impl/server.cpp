// ---------------------------------------------------------------------------
// OpenAI-compatible TTS HTTP server.
//
// Supported OpenAI-style endpoints:
//   POST /v1/audio/speech
//   POST /v1/audio/voice_consents
//   GET/PATCH/DELETE /v1/audio/voice_consents/{id}
//   POST /v1/audio/voices
//   GET/PATCH/DELETE /v1/audio/voices/{id}
//
// Local web management endpoints:
//   GET/POST /api/voices
//   GET/PATCH/DELETE /api/voices/{id}
//
// Voice metadata is kept in sqlite. Uploaded reference audio and native MIT2
// voice bundles are kept under the configured voice store directory.
// ---------------------------------------------------------------------------

#include "server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sqlite3.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mit2/bundle.hpp"
#include "mit2/metal_context.hpp"

using Clock = std::chrono::steady_clock;

std::string mtts_json_escape(const std::string& value);
void server_reset_tts_stage_acc();
bool server_run_tts_clone_native(const std::string& model_bundle_dir,
                                 const std::string& audio_wav,
                                 const std::string& output_voice_bundle);
bool server_run_tts_clone_fast(const std::string& audio_wav, const std::string& output_voice_bundle);
bool server_run_tts_product_entry(const std::string& command,
                           const std::string& bundle_dir,
                           const std::string& voice_bundle_dir,
                           const std::string& text,
                           const std::string& output_wav,
                           const std::string& preset);

namespace mit2_server {

struct ServerConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 3456;
    std::string model_bundle_dir;
    std::string voice_store_dir = "voices";
    uint32_t queue_size = 16;
    uint32_t tts_concurrency = 1;
    uint32_t clone_concurrency = 1;
    bool web_enabled = false;
    std::string web_key;
    bool verbose = false;
};

inline uint32_t env_u32(const char* name, uint32_t fallback, uint32_t min_value, uint32_t max_value) {
    if (const char* v = std::getenv(name)) {
        const long n = std::strtol(v, nullptr, 10);
        if (n >= static_cast<long>(min_value) && n <= static_cast<long>(max_value)) {
            return static_cast<uint32_t>(n);
        }
    }
    return fallback;
}

inline std::string env_string(const char* name, const std::string& fallback) {
    if (const char* v = std::getenv(name)) {
        if (*v) {
            return v;
        }
    }
    return fallback;
}

inline std::string now_epoch_string() {
    return std::to_string(static_cast<long long>(std::time(nullptr)));
}

inline std::string make_id(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    const auto n = counter.fetch_add(1);
    const auto t = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    std::ostringstream out;
    out << prefix << "_" << std::hex << t << n;
    return out.str();
}

inline std::string content_type_from_head(const std::string& head) {
    std::string lower = head;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    const size_t p = lower.find("content-type:");
    if (p == std::string::npos) {
        return {};
    }
    size_t start = p + 13;
    while (start < head.size() && std::isspace(static_cast<unsigned char>(head[start]))) {
        ++start;
    }
    size_t end = head.find("\r\n", start);
    if (end == std::string::npos) {
        end = head.size();
    }
    return head.substr(start, end - start);
}

inline std::string header_value(const std::string& head, const std::string& name) {
    std::string lower = head;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    std::string needle = name;
    std::transform(needle.begin(), needle.end(), needle.begin(), [](unsigned char c) { return std::tolower(c); });
    needle += ":";
    const size_t p = lower.find(needle);
    if (p == std::string::npos) {
        return {};
    }
    size_t start = p + needle.size();
    while (start < head.size() && std::isspace(static_cast<unsigned char>(head[start]))) {
        ++start;
    }
    size_t end = head.find("\r\n", start);
    if (end == std::string::npos) {
        end = head.size();
    }
    return head.substr(start, end - start);
}

inline std::string route_path_from_head(const std::string& head, std::string& method) {
    const size_t eol = head.find("\r\n");
    const std::string first = eol == std::string::npos ? head : head.substr(0, eol);
    const size_t sp1 = first.find(' ');
    const size_t sp2 = sp1 == std::string::npos ? std::string::npos : first.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        method.clear();
        return {};
    }
    method = first.substr(0, sp1);
    std::string path = first.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t q = path.find('?');
    if (q != std::string::npos) {
        path.resize(q);
    }
    return path;
}

inline std::string json_string_field(const std::string& body, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < body.size() && body[p] == ':') {
            ++p;
            while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
            if (p < body.size() && body[p] == '"') {
                ++p;
                std::string out;
                while (p < body.size() && body[p] != '"') {
                    char c = body[p];
                    if (c == '\\' && p + 1 < body.size()) {
                        const char e = body[p + 1];
                        p += 2;
                        switch (e) {
                            case 'n': out += '\n'; break;
                            case 'r': out += '\r'; break;
                            case 't': out += '\t'; break;
                            case 'b': out += '\b'; break;
                            case 'f': out += '\f'; break;
                            case 'u': {
                                if (p + 4 <= body.size()) {
                                    const unsigned cp = static_cast<unsigned>(std::strtoul(body.substr(p, 4).c_str(), nullptr, 16));
                                    p += 4;
                                    if (cp < 0x80) {
                                        out += static_cast<char>(cp);
                                    } else if (cp < 0x800) {
                                        out += static_cast<char>(0xC0 | (cp >> 6));
                                        out += static_cast<char>(0x80 | (cp & 0x3F));
                                    } else {
                                        out += static_cast<char>(0xE0 | (cp >> 12));
                                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                        out += static_cast<char>(0x80 | (cp & 0x3F));
                                    }
                                }
                                break;
                            }
                            default: out += e; break;
                        }
                    } else {
                        out += c;
                        ++p;
                    }
                }
                return out;
            }
        }
        pos = body.find(needle, pos + 1);
    }
    return {};
}

inline bool json_bool_field(const std::string& body, const std::string& key, bool fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    while (pos != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
        if (p < body.size() && body[p] == ':') {
            ++p;
            while (p < body.size() && std::isspace(static_cast<unsigned char>(body[p]))) ++p;
            if (body.compare(p, 4, "true") == 0) return true;
            if (body.compare(p, 5, "false") == 0) return false;
            return fallback;
        }
        pos = body.find(needle, pos + 1);
    }
    return fallback;
}

inline std::string json_voice_field(const std::string& body) {
    const std::string direct = json_string_field(body, "voice");
    if (!direct.empty()) {
        return direct;
    }
    const size_t p = body.find("\"voice\"");
    if (p == std::string::npos) {
        return {};
    }
    const size_t id_pos = body.find("\"id\"", p);
    if (id_pos == std::string::npos) {
        return {};
    }
    const size_t end_obj = body.find('}', p);
    if (end_obj != std::string::npos && id_pos > end_obj) {
        return {};
    }
    return json_string_field(body.substr(id_pos), "id");
}

struct MultipartPart {
    std::string name;
    std::string filename;
    std::string content_type;
    std::string data;
};

inline std::string trim_crlf(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
    return s;
}

inline std::vector<MultipartPart> parse_multipart(const std::string& content_type, const std::string& body) {
    std::vector<MultipartPart> parts;
    const std::string key = "boundary=";
    const size_t bp = content_type.find(key);
    if (bp == std::string::npos) {
        return parts;
    }
    std::string boundary = content_type.substr(bp + key.size());
    const size_t semicolon = boundary.find(';');
    if (semicolon != std::string::npos) {
        boundary.resize(semicolon);
    }
    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }
    const std::string marker = "--" + boundary;
    size_t pos = body.find(marker);
    while (pos != std::string::npos) {
        pos += marker.size();
        if (body.compare(pos, 2, "--") == 0) {
            break;
        }
        if (body.compare(pos, 2, "\r\n") == 0) {
            pos += 2;
        }
        const size_t header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) {
            break;
        }
        const std::string headers = body.substr(pos, header_end - pos);
        size_t data_start = header_end + 4;
        size_t next = body.find(marker, data_start);
        if (next == std::string::npos) {
            break;
        }
        std::string data = body.substr(data_start, next - data_start);
        data = trim_crlf(std::move(data));
        MultipartPart part;
        part.data = std::move(data);
        std::string lower = headers;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        const size_t cd = lower.find("content-disposition:");
        if (cd != std::string::npos) {
            const size_t line_end = lower.find("\r\n", cd);
            const std::string line = headers.substr(cd, (line_end == std::string::npos ? headers.size() : line_end) - cd);
            auto attr = [&](const std::string& k) -> std::string {
                const std::string needle = k + "=\"";
                const size_t a = line.find(needle);
                if (a == std::string::npos) return {};
                const size_t s = a + needle.size();
                const size_t e = line.find('"', s);
                return e == std::string::npos ? std::string{} : line.substr(s, e - s);
            };
            part.name = attr("name");
            part.filename = attr("filename");
        }
        const size_t ct = lower.find("content-type:");
        if (ct != std::string::npos) {
            size_t s = ct + 13;
            while (s < headers.size() && std::isspace(static_cast<unsigned char>(headers[s]))) ++s;
            size_t e = headers.find("\r\n", s);
            if (e == std::string::npos) e = headers.size();
            part.content_type = headers.substr(s, e - s);
        }
        if (!part.name.empty()) {
            parts.push_back(std::move(part));
        }
        pos = next;
    }
    return parts;
}

inline std::string multipart_value(const std::vector<MultipartPart>& parts, const std::string& name) {
    for (const auto& p : parts) {
        if (p.name == name) {
            return p.data;
        }
    }
    return {};
}

inline std::optional<MultipartPart> multipart_file(const std::vector<MultipartPart>& parts, const std::vector<std::string>& names) {
    for (const auto& n : names) {
        for (const auto& p : parts) {
            if (p.name == n && (!p.filename.empty() || !p.data.empty())) {
                return p;
            }
        }
    }
    return std::nullopt;
}

inline bool send_all(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        const ssize_t n = ::send(fd, data + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

inline void send_response(int fd, int code, const std::string& status,
                          const std::string& content_type, const std::string& body) {
    std::ostringstream head;
    head << "HTTP/1.1 " << code << " " << status << "\r\n"
         << "Content-Type: " << content_type << "\r\n"
         << "Content-Length: " << body.size() << "\r\n"
         << "Connection: close\r\n\r\n";
    const std::string h = head.str();
    send_all(fd, h.data(), h.size());
    send_all(fd, body.data(), body.size());
}

inline void send_json_error(int fd, int code, const std::string& status, const std::string& message) {
    send_response(fd, code, status, "application/json",
                  "{\"error\":{\"message\":\"" + mtts_json_escape(message) + "\"}}");
}

inline bool read_request(int fd, std::string& head, std::string& body) {
    std::string buf;
    char chunk[8192];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
        if (buf.size() > (1u << 20)) return false;
    }
    head = buf.substr(0, header_end);
    body = buf.substr(header_end + 4);
    size_t content_length = 0;
    std::string lower = head;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    const size_t cl = lower.find("content-length:");
    if (cl != std::string::npos) {
        content_length = static_cast<size_t>(std::strtoull(lower.c_str() + cl + 15, nullptr, 10));
    }
    if (content_length > (128u << 20)) return false;
    while (body.size() < content_length) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        body.append(chunk, static_cast<size_t>(n));
    }
    if (body.size() > content_length) {
        body.resize(content_length);
    }
    return true;
}

struct WorkEvent {
    uint64_t id = 0;
    std::string kind;
    std::string label;
    std::string status;
    std::string error;
    double elapsed_seconds = 0.0;
    std::string finished_at;
    double audio_seconds = 0.0;
    double rtf = 0.0;
};

class WorkDispatcher {
public:
    explicit WorkDispatcher(uint32_t max_waiting) : max_waiting_(std::max<uint32_t>(1, max_waiting)) {}

    bool submit(const std::string& kind,
                const std::string& label,
                const std::function<bool(std::string&, WorkEvent&)>& fn,
                std::string& error) {
        auto item = std::make_shared<Item>();
        item->id = next_id_.fetch_add(1) + 1;
        item->kind = kind;
        item->label = label;
        item->fn = fn;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() >= max_waiting_) {
                ++rejected_;
                error = kind + " queue is full";
                push_event_locked({item->id, kind, label, "rejected", error, 0.0, now_epoch_string(), 0.0, 0.0});
                return false;
            }
            queue_.push_back(item);
            ++submitted_;
            cv_.notify_one();
        }
        std::unique_lock<std::mutex> done_lock(item->mu);
        item->done_cv.wait(done_lock, [&] { return item->done; });
        error = item->error;
        return item->ok;
    }

    void run_forever() {
        for (;;) {
            std::shared_ptr<Item> item;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return !queue_.empty(); });
                item = queue_.front();
                queue_.pop_front();
                running_ = true;
                current_id_ = item->id;
                current_kind_ = item->kind;
                current_label_ = item->label;
                current_started_ = Clock::now();
            }
            std::string error;
            bool ok = false;
            WorkEvent event{item->id, item->kind, item->label, "", "", 0.0, "", 0.0, 0.0};
            const auto start = Clock::now();
            try {
                // Drain per request: this worker thread never returns, so any
                // Metal/MPS objects autoreleased during synthesis would leak
                // (resident memory climbs every request) without an explicit
                // pool. See mit2::AutoreleasePool.
                mit2::AutoreleasePool pool;
                ok = item->fn(error, event);
            } catch (const std::exception& e) {
                error = e.what();
                ok = false;
            }
            const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
            event.status = ok ? "completed" : "failed";
            event.error = error;
            event.elapsed_seconds = elapsed;
            event.finished_at = now_epoch_string();
            if (event.audio_seconds > 0.0) {
                event.rtf = elapsed / event.audio_seconds;
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                running_ = false;
                current_id_ = 0;
                current_kind_.clear();
                current_label_.clear();
                if (ok) {
                    ++completed_;
                } else {
                    ++failed_;
                }
                push_event_locked(event);
            }
            {
                std::lock_guard<std::mutex> done_lock(item->mu);
                item->ok = ok;
                item->error = error;
                item->done = true;
            }
            item->done_cv.notify_all();
        }
    }

    std::string status_json(const ServerConfig& cfg) {
        std::lock_guard<std::mutex> lock(mu_);
        std::ostringstream out;
        out << "{\"status\":\"ok\","
            << "\"model_bundle\":\"" << mtts_json_escape(cfg.model_bundle_dir) << "\","
            << "\"voice_store\":\"" << mtts_json_escape(cfg.voice_store_dir) << "\","
            << "\"web_enabled\":" << (cfg.web_enabled ? "true" : "false") << ","
            << "\"web_auth_required\":" << (cfg.web_enabled && !cfg.web_key.empty() ? "true" : "false") << ","
            << "\"configured_tts_concurrency\":" << cfg.tts_concurrency << ","
            << "\"configured_clone_concurrency\":" << cfg.clone_concurrency << ","
            << "\"effective_executor_concurrency\":1,"
            << "\"queue\":{\"max_waiting\":" << max_waiting_
            << ",\"waiting\":" << queue_.size()
            << ",\"running\":" << (running_ ? "true" : "false");
        if (running_) {
            const double elapsed = std::chrono::duration<double>(Clock::now() - current_started_).count();
            out << ",\"current\":{\"id\":" << current_id_
                << ",\"kind\":\"" << mtts_json_escape(current_kind_) << "\""
                << ",\"label\":\"" << mtts_json_escape(current_label_) << "\""
                << ",\"elapsed_seconds\":" << elapsed << "}";
        } else {
            out << ",\"current\":null";
        }
        out << "},\"totals\":{\"submitted\":" << submitted_
            << ",\"completed\":" << completed_
            << ",\"failed\":" << failed_
            << ",\"rejected\":" << rejected_ << "},\"recent\":[";
        for (size_t i = 0; i < recent_.size(); ++i) {
            if (i) out << ",";
            const auto& e = recent_[i];
            out << "{\"id\":" << e.id
                << ",\"kind\":\"" << mtts_json_escape(e.kind) << "\""
                << ",\"label\":\"" << mtts_json_escape(e.label) << "\""
                << ",\"status\":\"" << mtts_json_escape(e.status) << "\""
                << ",\"error\":\"" << mtts_json_escape(e.error) << "\""
                << ",\"elapsed_seconds\":" << e.elapsed_seconds
                << ",\"audio_seconds\":" << e.audio_seconds
                << ",\"rtf\":" << e.rtf
                << ",\"finished_at\":\"" << mtts_json_escape(e.finished_at) << "\"}";
        }
        out << "]}";
        return out.str();
    }

private:
    struct Item {
        uint64_t id = 0;
        std::string kind;
        std::string label;
        std::function<bool(std::string&, WorkEvent&)> fn;
        std::mutex mu;
        std::condition_variable done_cv;
        bool done = false;
        bool ok = false;
        std::string error;
    };

    void push_event_locked(const WorkEvent& event) {
        recent_.push_front(event);
        while (recent_.size() > 32) {
            recent_.pop_back();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Item>> queue_;
    std::deque<WorkEvent> recent_;
    std::atomic<uint64_t> next_id_{0};
    size_t max_waiting_ = 16;
    bool running_ = false;
    uint64_t current_id_ = 0;
    std::string current_kind_;
    std::string current_label_;
    Clock::time_point current_started_{};
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;
    uint64_t failed_ = 0;
    uint64_t rejected_ = 0;
};

struct VoiceRecord {
    std::string id;
    std::string name;
    std::string description;
    std::string bundle_path;
    std::string sample_path;
    double source_audio_seconds = 0.0;
    std::string source;
    std::string created_at;
    std::string updated_at;
};

struct ConsentRecord {
    std::string id;
    std::string name;
    std::string language;
    std::string recording_path;
    std::string created_at;
    std::string updated_at;
};

class VoiceStore {
public:
    explicit VoiceStore(std::string root) : root_(std::move(root)) {
        namespace fs = std::filesystem;
        fs::create_directories(root_);
        fs::create_directories(fs::path(root_) / "samples");
        fs::create_directories(fs::path(root_) / "bundles");
        const std::string db_path = (fs::path(root_) / "voices.sqlite").string();
        if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) {
            throw std::runtime_error("failed to open sqlite voice store: " + db_path);
        }
        exec("PRAGMA journal_mode=WAL");
        exec("CREATE TABLE IF NOT EXISTS voices ("
             "id TEXT PRIMARY KEY, name TEXT, description TEXT, bundle_path TEXT NOT NULL,"
             "sample_path TEXT, source_audio_seconds REAL DEFAULT 0, source TEXT,"
             "created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0)");
        exec_ignore_duplicate_column("ALTER TABLE voices ADD COLUMN source_audio_seconds REAL DEFAULT 0");
        exec("CREATE TABLE IF NOT EXISTS voice_consents ("
             "id TEXT PRIMARY KEY, name TEXT, language TEXT, recording_path TEXT,"
             "created_at TEXT, updated_at TEXT, deleted INTEGER DEFAULT 0)");
    }

    ~VoiceStore() {
        if (db_) {
            sqlite3_close(db_);
        }
    }

    std::string root() const { return root_; }

    std::string samples_dir() const { return (std::filesystem::path(root_) / "samples").string(); }
    std::string bundles_dir() const { return (std::filesystem::path(root_) / "bundles").string(); }

    void exec(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "sqlite error";
            sqlite3_free(err);
            throw std::runtime_error(message);
        }
    }

    void exec_ignore_duplicate_column(const std::string& sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
            std::string message = err ? err : "sqlite error";
            sqlite3_free(err);
            if (message.find("duplicate column name") == std::string::npos) {
                throw std::runtime_error(message);
            }
        }
    }

    void insert_voice(const VoiceRecord& v) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("INSERT OR REPLACE INTO voices"
                "(id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at,deleted)"
                " VALUES(?,?,?,?,?,?,?,?,?,0)", &st);
        bind(st, 1, v.id); bind(st, 2, v.name); bind(st, 3, v.description);
        bind(st, 4, v.bundle_path); bind(st, 5, v.sample_path);
        sqlite3_bind_double(st, 6, v.source_audio_seconds);
        bind(st, 7, v.source);
        bind(st, 8, v.created_at); bind(st, 9, v.updated_at);
        step_done(st);
    }

    std::optional<VoiceRecord> get_voice(const std::string& id_or_name) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at"
                " FROM voices WHERE deleted=0 AND (id=? OR name=?) LIMIT 1", &st);
        bind(st, 1, id_or_name);
        bind(st, 2, id_or_name);
        std::optional<VoiceRecord> out;
        if (sqlite3_step(st) == SQLITE_ROW) {
            out = row_voice(st);
        }
        sqlite3_finalize(st);
        return out;
    }

    std::vector<VoiceRecord> list_voices() {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,description,bundle_path,sample_path,source_audio_seconds,source,created_at,updated_at"
                " FROM voices WHERE deleted=0 ORDER BY created_at DESC", &st);
        std::vector<VoiceRecord> out;
        while (sqlite3_step(st) == SQLITE_ROW) {
            out.push_back(row_voice(st));
        }
        sqlite3_finalize(st);
        return out;
    }

    bool update_voice(const std::string& id, const VoiceRecord& patch) {
        auto current = get_voice(id);
        if (!current) return false;
        VoiceRecord v = *current;
        if (!patch.name.empty()) v.name = patch.name;
        if (!patch.description.empty()) v.description = patch.description;
        if (!patch.bundle_path.empty()) v.bundle_path = patch.bundle_path;
        if (!patch.sample_path.empty()) v.sample_path = patch.sample_path;
        if (patch.source_audio_seconds > 0.0) v.source_audio_seconds = patch.source_audio_seconds;
        if (!patch.source.empty()) v.source = patch.source;
        v.updated_at = now_epoch_string();
        insert_voice(v);
        return true;
    }

    bool delete_voice(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("UPDATE voices SET deleted=1, updated_at=? WHERE id=? AND deleted=0", &st);
        bind(st, 1, now_epoch_string());
        bind(st, 2, id);
        step_done(st);
        const bool changed = sqlite3_changes(db_) > 0;
        return changed;
    }

    void insert_consent(const ConsentRecord& c) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("INSERT OR REPLACE INTO voice_consents"
                "(id,name,language,recording_path,created_at,updated_at,deleted)"
                " VALUES(?,?,?,?,?,?,0)", &st);
        bind(st, 1, c.id); bind(st, 2, c.name); bind(st, 3, c.language);
        bind(st, 4, c.recording_path); bind(st, 5, c.created_at); bind(st, 6, c.updated_at);
        step_done(st);
    }

    std::optional<ConsentRecord> get_consent(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,language,recording_path,created_at,updated_at"
                " FROM voice_consents WHERE deleted=0 AND id=? LIMIT 1", &st);
        bind(st, 1, id);
        std::optional<ConsentRecord> out;
        if (sqlite3_step(st) == SQLITE_ROW) {
            out = row_consent(st);
        }
        sqlite3_finalize(st);
        return out;
    }

    std::vector<ConsentRecord> list_consents() {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("SELECT id,name,language,recording_path,created_at,updated_at"
                " FROM voice_consents WHERE deleted=0 ORDER BY created_at DESC", &st);
        std::vector<ConsentRecord> out;
        while (sqlite3_step(st) == SQLITE_ROW) {
            out.push_back(row_consent(st));
        }
        sqlite3_finalize(st);
        return out;
    }

    bool update_consent(const std::string& id, const ConsentRecord& patch) {
        auto current = get_consent(id);
        if (!current) return false;
        ConsentRecord c = *current;
        if (!patch.name.empty()) c.name = patch.name;
        if (!patch.language.empty()) c.language = patch.language;
        if (!patch.recording_path.empty()) c.recording_path = patch.recording_path;
        c.updated_at = now_epoch_string();
        insert_consent(c);
        return true;
    }

    bool delete_consent(const std::string& id) {
        std::lock_guard<std::mutex> lock(mu_);
        sqlite3_stmt* st = nullptr;
        prepare("UPDATE voice_consents SET deleted=1, updated_at=? WHERE id=? AND deleted=0", &st);
        bind(st, 1, now_epoch_string());
        bind(st, 2, id);
        step_done(st);
        return sqlite3_changes(db_) > 0;
    }

    std::string resolve_voice_bundle(const std::string& voice) {
        namespace fs = std::filesystem;
        if (voice.empty()) return {};
        if (auto rec = get_voice(voice)) {
            if (mit2::bundle_path_is_single_file(rec->bundle_path) ||
                fs::exists(fs::path(rec->bundle_path) / "manifest.json")) {
                return rec->bundle_path;
            }
        }
        if (mit2::bundle_path_is_single_file(voice)) return voice;
        if (fs::exists(fs::path(voice) / "manifest.json")) return voice;
        if (fs::is_regular_file(voice)) {
            std::string stem = voice;
            const size_t dot = stem.rfind(".pt");
            if (dot != std::string::npos) stem = stem.substr(0, dot);
            const std::string converted = stem + "_voice";
            if (mit2::bundle_path_is_single_file(converted) ||
                fs::exists(fs::path(converted) / "manifest.json")) {
                return converted;
            }
        }
        return {};
    }

private:
    void prepare(const std::string& sql, sqlite3_stmt** st) {
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, st, nullptr) != SQLITE_OK) {
            throw std::runtime_error(sqlite3_errmsg(db_));
        }
    }

    static void bind(sqlite3_stmt* st, int index, const std::string& value) {
        sqlite3_bind_text(st, index, value.c_str(), -1, SQLITE_TRANSIENT);
    }

    static std::string col(sqlite3_stmt* st, int index) {
        const unsigned char* v = sqlite3_column_text(st, index);
        return v ? reinterpret_cast<const char*>(v) : "";
    }

    static VoiceRecord row_voice(sqlite3_stmt* st) {
        return {col(st, 0), col(st, 1), col(st, 2), col(st, 3), col(st, 4),
                sqlite3_column_double(st, 5), col(st, 6), col(st, 7), col(st, 8)};
    }

    static ConsentRecord row_consent(sqlite3_stmt* st) {
        return {col(st, 0), col(st, 1), col(st, 2), col(st, 3), col(st, 4), col(st, 5)};
    }

    static void step_done(sqlite3_stmt* st) {
        const int rc = sqlite3_step(st);
        if (rc != SQLITE_DONE) {
            std::string msg = sqlite3_errmsg(sqlite3_db_handle(st));
            sqlite3_finalize(st);
            throw std::runtime_error(msg);
        }
        sqlite3_finalize(st);
    }

    std::string root_;
    sqlite3* db_ = nullptr;
    std::mutex mu_;
};

inline std::string voice_json(const VoiceRecord& v) {
    std::ostringstream out;
    out << "{\"id\":\"" << mtts_json_escape(v.id) << "\","
        << "\"object\":\"audio.voice\","
        << "\"name\":\"" << mtts_json_escape(v.name) << "\","
        << "\"description\":\"" << mtts_json_escape(v.description) << "\","
        << "\"bundle_path\":\"" << mtts_json_escape(v.bundle_path) << "\","
        << "\"sample_path\":\"" << mtts_json_escape(v.sample_path) << "\","
        << "\"source_audio_seconds\":" << v.source_audio_seconds << ","
        << "\"source\":\"" << mtts_json_escape(v.source) << "\","
        << "\"created_at\":\"" << mtts_json_escape(v.created_at) << "\","
        << "\"updated_at\":\"" << mtts_json_escape(v.updated_at) << "\"}";
    return out.str();
}

inline std::string consent_json(const ConsentRecord& c, const std::string& voice_id = "") {
    std::ostringstream out;
    out << "{\"id\":\"" << mtts_json_escape(c.id) << "\","
        << "\"object\":\"audio.voice_consent\","
        << "\"name\":\"" << mtts_json_escape(c.name) << "\","
        << "\"language\":\"" << mtts_json_escape(c.language) << "\","
        << "\"recording_path\":\"" << mtts_json_escape(c.recording_path) << "\",";
    if (!voice_id.empty()) {
        out << "\"voice_id\":\"" << mtts_json_escape(voice_id) << "\",";
    }
    out << "\"created_at\":\"" << mtts_json_escape(c.created_at) << "\","
        << "\"updated_at\":\"" << mtts_json_escape(c.updated_at) << "\"}";
    return out.str();
}

inline bool web_key_authorized(const ServerConfig& cfg, const std::string& head, const std::string& body) {
    if (!cfg.web_enabled || cfg.web_key.empty()) {
        return true;
    }
    const std::string header_key = header_value(head, "x-mtts-web-key");
    if (header_key == cfg.web_key) {
        return true;
    }
    const std::string auth = header_value(head, "authorization");
    const std::string prefix = "Bearer ";
    if (auth.size() > prefix.size() && auth.compare(0, prefix.size(), prefix) == 0 &&
        auth.substr(prefix.size()) == cfg.web_key) {
        return true;
    }
    return json_string_field(body, "key") == cfg.web_key;
}

inline const char* web_index_html() {
    return R"MTTSWEB(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MTTS Admin</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1c2430;background:#f7f8fa}
body{margin:0}
button,input,textarea{font:inherit}
.top{height:56px;background:#111827;color:white;display:flex;align-items:center;justify-content:space-between;padding:0 20px}
.brand{font-weight:700;letter-spacing:.02em}.wrap{max-width:1180px;margin:0 auto;padding:20px}
.tabs{display:flex;gap:8px;margin-bottom:16px}.tabs button{border:1px solid #d6dbe3;background:white;padding:8px 12px;border-radius:6px;cursor:pointer}.tabs button.active{background:#0f766e;color:white;border-color:#0f766e}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.voice-layout{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}.voice-layout.testing{grid-template-columns:minmax(0,1fr) 380px}.panel{background:white;border:1px solid #dfe3ea;border-radius:8px;padding:16px}.panel h2{margin:0 0 12px;font-size:18px}.panel h3{margin:16px 0 8px;font-size:15px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.field{display:grid;gap:6px;margin-bottom:10px}.field label{font-size:12px;color:#5b6472}.field input,.field textarea{border:1px solid #cfd6e1;border-radius:6px;padding:8px;background:white}.field textarea{min-height:64px}
	.primary{background:#0f766e;color:white;border:0;border-radius:6px;padding:8px 12px;cursor:pointer}.danger{background:#b42318;color:white;border:0;border-radius:6px;padding:7px 10px;cursor:pointer}.secondary{background:#eef2f7;color:#1c2430;border:0;border-radius:6px;padding:7px 10px;cursor:pointer}
	table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e5e9f0;padding:8px;font-size:13px;vertical-align:top}th{color:#5b6472;font-weight:600}.muted{color:#667085;font-size:13px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}
		.icon-cell{width:72px}.select-cell{width:32px}.hover-actions{display:flex;gap:6px;opacity:0;transition:opacity .12s}.voice-row:hover .hover-actions{opacity:1}.icon-btn{width:28px;height:28px;border:1px solid #d6dbe3;background:white;border-radius:6px;cursor:pointer;line-height:1}.icon-btn.active{background:#0f766e;color:white;border-color:#0f766e}.icon-btn.danger-icon{background:#fff5f4;color:#b42318;border-color:#f5b5ae}.icon-btn[disabled]{opacity:.45;cursor:not-allowed}.voice-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.voice-head h2{margin:0}.voice-tools{display:flex;gap:6px;margin-left:auto}.edit-cell{position:relative;min-height:28px}.edit-value{padding-right:32px;white-space:pre-wrap}.edit-btn,.copy-btn{opacity:0;transition:opacity .12s}.voice-row:hover .edit-btn,.path-row:hover .copy-btn{opacity:1}.edit-btn{position:absolute;right:0;top:0}.edit-input{width:100%;box-sizing:border-box;border:1px solid #cfd6e1;border-radius:6px;padding:6px;background:white}.bundle-path{display:flex;align-items:flex-start;gap:6px}.bundle-path .mono{flex:1}.copy-btn{flex:0 0 auto}.add-voice{margin-top:16px;border-top:1px solid #e5e9f0;padding-top:12px}.add-voice summary{cursor:pointer;font-weight:700}.add-voice-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px;align-items:start}.add-voice-grid h3{margin-top:16px}.file-row{display:flex;gap:8px;align-items:center}.file-row input[type=file]{flex:1;min-width:0}.local-preview{width:100%;margin-top:8px}.side-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.side-head h2{margin:0}.duration{white-space:nowrap}.voice-meta{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px 12px;margin:12px 0 14px}.voice-meta dt{color:#5b6472;font-size:12px}.voice-meta dd{margin:0;font-size:13px;word-break:break-word}.switch-row{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:2px 0 12px}.switch-label{font-size:13px;color:#1c2430}.android-switch{position:relative;display:inline-flex;align-items:center;width:46px;height:28px;cursor:pointer;flex:0 0 auto}.android-switch input{position:absolute;opacity:0;width:1px;height:1px}.switch-track{width:46px;height:28px;border-radius:999px;background:#cfd6e1;transition:background .16s,box-shadow .16s}.switch-track:after{content:"";position:absolute;top:3px;left:3px;width:22px;height:22px;border-radius:50%;background:white;box-shadow:0 1px 3px rgba(16,24,40,.28);transition:transform .16s}.android-switch input:checked+.switch-track{background:#0f766e}.android-switch input:checked+.switch-track:after{transform:translateX(18px)}.android-switch input:focus-visible+.switch-track{box-shadow:0 0 0 3px rgba(15,118,110,.22)}.speech-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}.speech-metric{background:#f4f6f8;border:1px solid #e2e6ed;border-radius:6px;padding:8px}.speech-metric span{display:block;color:#667085;font-size:12px}.speech-metric b{font-size:16px}button[disabled]{opacity:.65;cursor:not-allowed}.spinner{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.45);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;margin-right:6px;vertical-align:-1px}@keyframes spin{to{transform:rotate(360deg)}}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.metric{border:1px solid #e2e6ed;border-radius:6px;padding:10px;background:#f4f6f8}.metric b{display:block;font-size:20px}.metric-waiting{background:#eef4ff;border-color:#b2ccff}.metric-running{background:#ecfdf3;border-color:#75e0a7}.metric-idle{background:#f4f6f8;border-color:#d0d5dd}.metric-submitted{background:#fef7c3;border-color:#fdb022}.metric-failed{background:#fef3f2;border-color:#fda29b}.badge{display:inline-flex;align-items:center;border-radius:999px;padding:3px 8px;font-size:12px;font-weight:700}.badge-tts{background:#e0f2fe;color:#075985}.badge-clone{background:#ecfdf3;color:#067647}.badge-running{background:#dcfce7;color:#166534}.badge-completed{background:#d1fae5;color:#065f46}.badge-failed,.badge-rejected{background:#fee2e2;color:#991b1b}.badge-time{background:#fef3c7;color:#92400e}.job-card{border:1px solid #d6dbe3;border-radius:8px;padding:12px;background:#fff}.job-top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.job-label{font-size:14px;line-height:1.45;word-break:break-word}.hidden{display:none!important}.login{max-width:380px;margin:80px auto;background:white;border:1px solid #dfe3ea;border-radius:8px;padding:20px}
.toast{position:fixed;right:16px;bottom:16px;background:#111827;color:white;padding:10px 12px;border-radius:6px;max-width:520px;box-shadow:0 8px 30px rgba(0,0,0,.18)}
@media(max-width:850px){.grid,.voice-layout.testing,.add-voice-grid{grid-template-columns:1fr}.status{grid-template-columns:1fr 1fr}.top{padding:0 12px}.wrap{padding:12px}.hover-actions,.edit-btn,.copy-btn{opacity:1}}
</style>
</head>
<body>
<div class="top"><div class="brand">MTTS Admin</div><div id="serverLine" class="muted"></div></div>
<div id="login" class="login hidden">
  <h2>Admin Login</h2>
  <p class="muted">Enter the web key configured with <span class="mono">--webkey</span>.</p>
  <div class="field"><label>Web key</label><input id="loginKey" type="password" autocomplete="current-password"></div>
  <button class="primary" onclick="login()">Login</button>
</div>
<div id="app" class="wrap hidden">
  <div class="tabs">
    <button id="tabStatus" class="active" onclick="showTab('status')">Status</button>
    <button id="tabVoices" onclick="showTab('voices')">Voices</button>
    <button class="secondary" onclick="logout()">Logout</button>
  </div>
  <section id="pageStatus">
    <div class="panel">
      <h2>Runtime Status</h2>
      <div id="metrics" class="status"></div>
      <div id="currentJobSection" class="hidden">
        <h3>Current Job</h3>
        <div id="currentJob"></div>
      </div>
      <h3>Recent Jobs</h3>
      <div id="recentJobs"></div>
    </div>
  </section>
  <section id="pageVoices" class="hidden">
    <div id="voiceLayout" class="voice-layout">
      <div class="panel">
	        <div class="voice-head"><h2>Voices</h2><div class="voice-tools"><button class="icon-btn" title="Refresh voices" onclick="loadVoices()">&#8635;</button><button id="voiceManageToggle" class="icon-btn" title="Manage voices" onclick="toggleVoiceManage()">&#9745;</button><button id="voiceBatchDelete" class="icon-btn danger-icon hidden" title="Delete selected voices" onclick="deleteSelectedVoices()" disabled>&#128465;</button></div></div>
        <div id="voicesTable" style="margin-top:12px"></div>
        <audio id="previewAudio" controls class="hidden" style="width:100%;margin-top:12px"></audio>
        <details class="add-voice">
          <summary>Add Voice</summary>
          <div class="add-voice-grid">
            <div>
              <h3>Create Voice</h3>
              <div class="field"><label>Name</label><input id="cloneName" placeholder="demo voice"></div>
              <div class="field"><label>Description</label><input id="cloneDescription" placeholder="created from audio"></div>
	              <div class="field"><label>Audio sample</label><div class="file-row"><input id="cloneFile" type="file" accept="audio/*,.wav" onchange="onCloneFileChange()"><button id="clonePreviewBtn" class="icon-btn" title="Preview selected audio" onclick="previewCloneFile()" disabled>&#9658;</button></div><audio id="clonePreviewAudio" controls class="hidden local-preview"></audio></div>
	              <button id="cloneSubmit" class="primary" onclick="cloneVoice()">Create Voice</button>
            </div>
            <div>
              <h3>Import Existing Bundle</h3>
              <div class="field"><label>Name</label><input id="importName" placeholder="qin"></div>
              <div class="field"><label>Description</label><input id="importDescription" placeholder="local voice"></div>
              <div class="field"><label>Bundle path</label><input id="importBundle" placeholder="sample/qin.pt"></div>
	              <button id="importSubmit" class="primary" onclick="importVoice()">Import</button>
            </div>
          </div>
        </details>
      </div>
      <aside id="speechPanel" class="panel hidden">
      <div class="side-head"><h2>TTS Speech</h2><button class="secondary" onclick="closeSpeechPanel()">Close</button></div>
      <dl class="voice-meta">
        <dt>Voice ID</dt><dd id="speechVoiceId" class="mono">-</dd>
        <dt>Name</dt><dd id="speechVoiceName">-</dd>
        <dt>Description</dt><dd id="speechVoiceDescription">-</dd>
        <dt>Bundle</dt><dd class="path-row"><div class="bundle-path"><span id="speechBundlePath" class="mono">-</span><button class="icon-btn copy-btn" title="Copy bundle path" onclick="copyBundlePath()">&#10697;</button></div></dd>
      </dl>
      <div class="field"><label>Text</label><textarea id="speechText">你好世界</textarea></div>
      <div class="switch-row">
        <span class="switch-label">Auto play after generation</span>
        <label class="android-switch" title="Auto play generated audio">
          <input id="speechAutoPlay" type="checkbox" onchange="setSpeechAutoPlay(this.checked)">
          <span class="switch-track"></span>
        </label>
      </div>
      <button id="speechGenerate" class="primary" onclick="speak()">Generate WAV</button>
      <p id="speechResult" class="muted"></p>
      <div id="speechMetrics" class="speech-metrics hidden">
        <div class="speech-metric"><span>Audio</span><b id="speechAudioSeconds">-</b></div>
        <div class="speech-metric"><span>Elapsed</span><b id="speechElapsedSeconds">-</b></div>
        <div class="speech-metric"><span>RTF</span><b id="speechRtf">-</b></div>
      </div>
      <audio id="audio" controls class="hidden" style="width:100%;margin-top:12px"></audio>
      </aside>
    </div>
  </section>
</div>
<div id="toast" class="toast hidden"></div>
<script>
		let key=localStorage.getItem('mttsWebKey')||'';let poll=null;let selectedSpeechVoice='';let voiceById={};let authRequired=false;let clonePreviewUrl='';let speechAudioUrl='';let previewAudioUrl='';let voiceManageMode=false;let selectedVoiceIds=new Set();let speechAutoPlay=localStorage.getItem('mttsSpeechAutoPlay')!=='0';
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.remove('hidden');setTimeout(()=>t.classList.add('hidden'),4500)}
async function api(path,opt={}){opt.headers=opt.headers||{};if(key)opt.headers['X-MTTS-Web-Key']=key;if(opt.json){opt.headers['Content-Type']='application/json';opt.body=JSON.stringify(opt.json);delete opt.json}const r=await fetch('/web/api'+path,opt);if(r.status===401){const e=new Error('unauthorized');e.status=401;showLogin();throw e}if(!r.ok){let m=await r.text();const e=new Error(m);e.status=r.status;throw e}const ct=r.headers.get('content-type')||'';return ct.includes('application/json')?r.json():r.blob()}
function showLogin(){document.getElementById('login').classList.remove('hidden');document.getElementById('app').classList.add('hidden');if(poll)clearInterval(poll)}
function showApp(){document.getElementById('login').classList.add('hidden');document.getElementById('app').classList.remove('hidden');initSpeechAutoPlay();loadStatus();loadVoices();if(poll)clearInterval(poll);poll=setInterval(loadStatus,1000)}
async function login(){key=document.getElementById('loginKey').value;try{await api('/login',{method:'POST',json:{key}});localStorage.setItem('mttsWebKey',key);showApp()}catch(e){toast('Login failed')}}
function logout(){localStorage.removeItem('mttsWebKey');key='';if(authRequired)showLogin();else showApp()}
function showTab(n){['Status','Voices'].forEach(x=>{document.getElementById('tab'+x).classList.toggle('active',x.toLowerCase()===n);document.getElementById('page'+x).classList.toggle('hidden',x.toLowerCase()!==n)})}
async function loadStatus(){try{const s=await api('/status');authRequired=!!s.web_auth_required;document.getElementById('serverLine').textContent=s.model_bundle+' | '+s.voice_store;document.getElementById('metrics').innerHTML=[
metricHtml('Waiting',s.queue.waiting,'waiting'),metricHtml('Running',s.queue.running?'yes':'no',s.queue.running?'running':'idle'),metricHtml('Submitted',s.totals.submitted,'submitted'),metricHtml('Failed',s.totals.failed,'failed')
].join('');const current=document.getElementById('currentJobSection');if(s.queue.current){current.classList.remove('hidden');document.getElementById('currentJob').innerHTML=currentJobHtml(s.queue.current)}else{current.classList.add('hidden');document.getElementById('currentJob').innerHTML=''}document.getElementById('recentJobs').innerHTML=recentJobsHtml(s.recent)}catch(e){if(e.status===401)return;document.getElementById('metrics').innerHTML='<div class="metric metric-failed"><span class="muted">Status</span><b>Error</b></div>';document.getElementById('recentJobs').innerHTML='<p class="muted">'+escapeHtml(e.message||'failed to load status')+'</p>'}}
function metricHtml(label,value,tone){return `<div class="metric metric-${tone}"><span class="muted">${label}</span><b>${value}</b></div>`}
function currentJobHtml(j){return `<div class="job-card"><div class="job-top"><span class="badge badge-${escapeAttr(j.kind)}">${escapeHtml(j.kind)}</span><span class="badge badge-running">running</span><span class="badge badge-time">${formatHms(j.elapsed_seconds)}</span><span class="muted">#${j.id}</span></div><div class="job-label">${escapeHtml(truncateText(j.label||'',120))}</div></div>`}
function recentJobsHtml(jobs){if(!jobs.length)return '<p class="muted">No recent jobs</p>';return '<table><thead><tr><th>ID</th><th>Kind</th><th>Status</th><th>Elapsed</th><th>RTF</th><th>Label</th></tr></thead><tbody>'+jobs.map(j=>`<tr><td>${j.id}</td><td><span class="badge badge-${escapeAttr(j.kind)}">${escapeHtml(j.kind)}</span></td><td><span class="badge badge-${escapeAttr(j.status)}">${escapeHtml(j.status)}</span></td><td>${Number(j.elapsed_seconds).toFixed(2)}s</td><td>${j.rtf?Number(j.rtf).toFixed(2):'-'}</td><td>${escapeHtml(truncateText(j.error||j.label||'',90))}</td></tr>`).join('')+'</tbody></table>'}
async function loadVoices(){try{const v=await api('/voices');voiceById=Object.fromEntries(v.data.map(x=>[x.id,x]));selectedVoiceIds=new Set([...selectedVoiceIds].filter(id=>voiceById[id]));const selectHead=voiceManageMode?'<th class="select-cell"></th>':'';document.getElementById('voicesTable').innerHTML='<table><thead><tr>'+selectHead+'<th>Name</th><th>Description</th><th>Duration</th><th class="icon-cell"></th></tr></thead><tbody>'+v.data.map(voiceRowHtml).join('')+'</tbody></table>';updateVoiceManageUi()}catch(e){toast(e.message)}}
function voiceRowHtml(x){const selectCell=voiceManageMode?`<td class="select-cell"><input type="checkbox" aria-label="Select ${escapeAttr(x.name||x.id)}" onchange="toggleVoiceSelected('${escapeJs(x.id)}',this.checked)" ${selectedVoiceIds.has(x.id)?'checked':''}></td>`:'';return `<tr class="voice-row">${selectCell}<td>${editableCell(x,'name')}</td><td>${editableCell(x,'description')}</td><td class="duration">${formatSeconds(x.source_audio_seconds)}</td><td class="icon-cell"><div class="hover-actions"><button class="icon-btn" title="Preview source audio" onclick="previewSourceAudio('${escapeJs(x.id)}')">&#9658;</button><button class="icon-btn" title="Test TTS speech" onclick="openSpeechPanel('${escapeJs(x.id)}')">&#9835;</button></div></td></tr>`}
function editableCell(x,field){const id=escapeJs(x.id);const value=escapeHtml(x[field]||'');const label=field==='name'?'name':'description';return `<div id="cell-${label}-${escapeAttr(x.id)}" class="edit-cell"><div class="edit-value">${value||'-'}</div><button class="icon-btn edit-btn" title="Edit ${label}" onclick="beginEdit('${id}','${label}')">&#9998;</button></div>`}
	async function importVoice(){const btn=document.getElementById('importSubmit');try{setButtonBusy(btn,true,'Importing');await api('/voices',{method:'POST',json:{name:val('importName'),description:val('importDescription'),bundle_path:val('importBundle')}});toast('Voice imported');loadVoices()}catch(e){toast(e.message)}finally{setButtonBusy(btn,false)}}
	async function cloneVoice(){const btn=document.getElementById('cloneSubmit');try{const f=document.getElementById('cloneFile').files[0];if(!f){toast('Choose an audio file');return}setButtonBusy(btn,true,'Creating');const fd=new FormData();fd.append('name',val('cloneName'));fd.append('description',val('cloneDescription'));fd.append('audio_sample',f);await api('/voices',{method:'POST',body:fd});toast('Clone queued/completed');loadVoices();loadStatus()}catch(e){toast(e.message)}finally{setButtonBusy(btn,false)}}
	function setButtonBusy(btn,busy,label){if(!btn)return;if(!btn.dataset.idleHtml)btn.dataset.idleHtml=btn.innerHTML;btn.disabled=busy;btn.innerHTML=busy?`<span class="spinner"></span>${escapeHtml(label||'Working')}`:btn.dataset.idleHtml}
	function onCloneFileChange(){const input=document.getElementById('cloneFile');const f=input.files&&input.files[0];const btn=document.getElementById('clonePreviewBtn');const audio=document.getElementById('clonePreviewAudio');if(clonePreviewUrl){URL.revokeObjectURL(clonePreviewUrl);clonePreviewUrl=''}audio.pause();audio.removeAttribute('src');audio.classList.add('hidden');if(!f){btn.disabled=true;return}document.getElementById('cloneName').value=fileStem(f.name);clonePreviewUrl=URL.createObjectURL(f);btn.disabled=false}
	function previewCloneFile(){if(!clonePreviewUrl)return;const a=document.getElementById('clonePreviewAudio');a.src=clonePreviewUrl;a.classList.remove('hidden');a.play().catch(()=>{})}
	function fileStem(name){const base=String(name||'').split(/[\\/]/).pop()||'';const dot=base.lastIndexOf('.');return dot>0?base.slice(0,dot):base}
async function saveVoiceField(id,field,value){try{const patch={};patch[field]=value;const updated=await api('/voices/'+id,{method:'PATCH',json:patch});voiceById[id]=updated;toast('Saved');loadVoices();if(selectedSpeechVoice===id)openSpeechPanel(id)}catch(e){toast(e.message)}}
function beginEdit(id,field){const voice=voiceById[id];if(!voice)return;const cell=document.getElementById(`cell-${field}-${id}`);if(!cell)return;const current=voice[field]||'';cell.innerHTML=`<input class="edit-input" id="edit-${field}-${id}" value="${escapeAttr(current)}" onkeydown="editKey(event,'${escapeJs(id)}','${field}')">`;const input=document.getElementById(`edit-${field}-${id}`);input.focus();input.select()}
function editKey(event,id,field){if(event.key==='Enter'){event.preventDefault();saveVoiceField(id,field,event.target.value)}else if(event.key==='Escape'){event.preventDefault();loadVoices()}}
async function deleteVoice(id){if(!confirm('Delete '+id+'?'))return;try{await api('/voices/'+id,{method:'DELETE'});toast('Deleted');loadVoices()}catch(e){toast(e.message)}}
function toggleVoiceManage(){voiceManageMode=!voiceManageMode;if(!voiceManageMode)selectedVoiceIds.clear();loadVoices()}
function toggleVoiceSelected(id,checked){if(checked)selectedVoiceIds.add(id);else selectedVoiceIds.delete(id);updateVoiceManageUi()}
function updateVoiceManageUi(){const manage=document.getElementById('voiceManageToggle');const del=document.getElementById('voiceBatchDelete');if(manage)manage.classList.toggle('active',voiceManageMode);if(del){del.classList.toggle('hidden',!voiceManageMode);del.disabled=selectedVoiceIds.size===0;del.title=selectedVoiceIds.size?`Delete ${selectedVoiceIds.size} selected voice${selectedVoiceIds.size>1?'s':''}`:'Delete selected voices'}}
async function deleteSelectedVoices(){const ids=[...selectedVoiceIds];if(!ids.length)return;if(!confirm('Delete '+ids.length+' selected voice'+(ids.length>1?'s':'')+'?'))return;const btn=document.getElementById('voiceBatchDelete');try{btn.disabled=true;for(const id of ids){await api('/voices/'+id,{method:'DELETE'})}selectedVoiceIds.clear();toast('Deleted '+ids.length+' voice'+(ids.length>1?'s':''));loadVoices()}catch(e){toast(e.message);updateVoiceManageUi()}}
async function previewSourceAudio(id){try{const blob=await api('/voices/'+id+'/source-audio');if(previewAudioUrl){URL.revokeObjectURL(previewAudioUrl);previewAudioUrl=''}previewAudioUrl=URL.createObjectURL(blob);const a=document.getElementById('previewAudio');a.src=previewAudioUrl;a.classList.remove('hidden');await a.play().catch(()=>{})}catch(e){toast(e.message)}}
function openSpeechPanel(id){const voice=voiceById[id]||{id,name:'-',description:'-',bundle_path:'-'};selectedSpeechVoice=id;document.getElementById('speechVoiceId').textContent=id;document.getElementById('speechVoiceName').textContent=voice.name||'-';document.getElementById('speechVoiceDescription').textContent=voice.description||'-';document.getElementById('speechBundlePath').textContent=voice.bundle_path||'-';document.getElementById('voiceLayout').classList.add('testing');document.getElementById('speechPanel').classList.remove('hidden');document.getElementById('speechResult').textContent='';document.getElementById('speechMetrics').classList.add('hidden')}
function closeSpeechPanel(){selectedSpeechVoice='';document.getElementById('speechPanel').classList.add('hidden');document.getElementById('voiceLayout').classList.remove('testing')}
async function copyBundlePath(){const text=document.getElementById('speechBundlePath').textContent;if(!text||text==='-')return;try{await navigator.clipboard.writeText(text);toast('Bundle path copied')}catch(e){toast(text)}}
function initSpeechAutoPlay(){const input=document.getElementById('speechAutoPlay');if(input)input.checked=speechAutoPlay}
function setSpeechAutoPlay(enabled){speechAutoPlay=!!enabled;localStorage.setItem('mttsSpeechAutoPlay',speechAutoPlay?'1':'0')}
async function speak(){
const btn=document.getElementById('speechGenerate');const old=btn.innerHTML;
try{
if(!selectedSpeechVoice){toast('Select a voice');return}
btn.disabled=true;
btn.innerHTML='<span class="spinner"></span>Generating';
document.getElementById('speechResult').textContent='Synthesizing audio...';
document.getElementById('speechMetrics').classList.add('hidden');
const start=performance.now();
const body={model:'mtts',input:val('speechText'),voice:selectedSpeechVoice,response_format:'wav'};
const blob=await api('/speech',{method:'POST',json:body});
const elapsed=(performance.now()-start)/1000;
const audioSeconds=await wavDurationSeconds(blob);
const rtf=audioSeconds>0?elapsed/audioSeconds:0;
if(speechAudioUrl){URL.revokeObjectURL(speechAudioUrl);speechAudioUrl=''}
speechAudioUrl=URL.createObjectURL(blob);
const a=document.getElementById('audio');
a.src=speechAudioUrl;
a.classList.remove('hidden');
document.getElementById('speechResult').textContent='Generated '+Math.round(blob.size/1024)+' KB';
document.getElementById('speechAudioSeconds').textContent=formatSeconds(audioSeconds);
document.getElementById('speechElapsedSeconds').textContent=elapsed.toFixed(2)+'s';
document.getElementById('speechRtf').textContent=rtf>0?rtf.toFixed(2):'-';
document.getElementById('speechMetrics').classList.remove('hidden');
if(speechAutoPlay)a.play().catch(()=>{});
}catch(e){toast(e.message);document.getElementById('speechResult').textContent='Generation failed'}
finally{btn.disabled=false;btn.innerHTML=old}
}
function val(id){return document.getElementById(id).value}
function formatSeconds(v){const n=Number(v||0);if(!Number.isFinite(n)||n<=0)return '-';return n<60?n.toFixed(1)+'s':Math.floor(n/60)+'m '+Math.round(n%60)+'s'}
function formatHms(v){const n=Math.max(0,Math.floor(Number(v||0)));const h=String(Math.floor(n/3600)).padStart(2,'0');const m=String(Math.floor((n%3600)/60)).padStart(2,'0');const s=String(n%60).padStart(2,'0');return h+':'+m+':'+s}
function truncateText(s,max){const chars=Array.from(String(s||''));return chars.length>max?chars.slice(0,max).join('')+'...':chars.join('')}
async function wavDurationSeconds(blob){const buf=await blob.arrayBuffer();const view=new DataView(buf);if(buf.byteLength<44||text4(view,0)!=='RIFF'||text4(view,8)!=='WAVE')return 0;let pos=12;let channels=0,sampleRate=0,bits=0,dataBytes=0;while(pos+8<=buf.byteLength){const id=text4(view,pos);const size=view.getUint32(pos+4,true);const data=pos+8;if(data+size>buf.byteLength)break;if(id==='fmt '&&size>=16){channels=view.getUint16(data+2,true);sampleRate=view.getUint32(data+4,true);bits=view.getUint16(data+14,true)}else if(id==='data'){dataBytes=size}pos=data+size+(size&1)}const bytesPerSecond=sampleRate*channels*bits/8;return bytesPerSecond>0?dataBytes/bytesPerSecond:0}
function text4(view,pos){return String.fromCharCode(view.getUint8(pos),view.getUint8(pos+1),view.getUint8(pos+2),view.getUint8(pos+3))}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function escapeAttr(s){return escapeHtml(s).replace(/`/g,'&#96;')}
function escapeJs(s){return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'")}
(async()=>{try{const s=await api('/status');authRequired=!!s.web_auth_required;showApp()}catch(e){if(e.status===401)showLogin();else{showApp();toast(e.message||'Status unavailable')}}})();
</script>
</body>
</html>)MTTSWEB";
}

inline void save_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write file: " + path);
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

inline bool path_is_under_dir(const std::filesystem::path& path, const std::filesystem::path& dir) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path canonical_path = fs::weakly_canonical(path, ec);
    if (ec) return false;
    const fs::path canonical_dir = fs::weakly_canonical(dir, ec);
    if (ec) return false;
    auto dir_it = canonical_dir.begin();
    auto path_it = canonical_path.begin();
    for (; dir_it != canonical_dir.end(); ++dir_it, ++path_it) {
        if (path_it == canonical_path.end() || *path_it != *dir_it) {
            return false;
        }
    }
    return path_it != canonical_path.end();
}

inline bool remove_path_under_dir(const std::string& path, const std::string& dir, const char* label) {
    namespace fs = std::filesystem;
    if (path.empty() || dir.empty()) return false;
    const fs::path p(path);
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) return false;
    if (!path_is_under_dir(p, fs::path(dir))) return false;
    if (fs::is_directory(p, ec)) {
        fs::remove_all(p, ec);
    } else {
        fs::remove(p, ec);
    }
    if (ec) {
        std::cerr << "warning: failed to delete " << label << ": " << path
                  << " (" << ec.message() << ")" << std::endl;
        return false;
    }
    return true;
}

inline std::string safe_ext(const std::string& filename, const std::string& fallback) {
    const size_t dot = filename.rfind('.');
    if (dot == std::string::npos || dot + 1 >= filename.size()) return fallback;
    std::string ext = filename.substr(dot);
    if (ext.size() > 8) return fallback;
    for (char c : ext) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.')) return fallback;
    }
    return ext;
}

inline uint16_t read_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline std::string read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline double wav_duration_seconds_from_bytes(const std::string& bytes) {
    if (bytes.size() < 44 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) {
        return 0.0;
    }
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint32_t data_bytes = 0;
    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const std::string chunk_id(bytes.data() + pos, bytes.data() + pos + 4);
        const uint32_t chunk_size = read_le32(data + pos + 4);
        const size_t chunk_data = pos + 8;
        if (chunk_data + chunk_size > bytes.size()) break;
        if (chunk_id == "fmt " && chunk_size >= 16) {
            channels = read_le16(data + chunk_data + 2);
            sample_rate = read_le32(data + chunk_data + 4);
            bits_per_sample = read_le16(data + chunk_data + 14);
        } else if (chunk_id == "data") {
            data_bytes = chunk_size;
        }
        pos = chunk_data + chunk_size + (chunk_size & 1u);
    }
    const double bytes_per_second = static_cast<double>(sample_rate) *
        static_cast<double>(channels) * static_cast<double>(bits_per_sample) / 8.0;
    return bytes_per_second > 0.0 ? static_cast<double>(data_bytes) / bytes_per_second : 0.0;
}

inline double wav_duration_seconds_from_file(const std::string& path) {
    return wav_duration_seconds_from_bytes(read_file_bytes(path));
}

inline std::string source_audio_bytes_from_bundle(const std::string& bundle_path) {
    try {
        mit2::Bundle bundle(bundle_path);
        const auto* info = bundle.find("source_audio_wav_bytes");
        if (!info || info->dtype != "u8" || info->nbytes == 0) return {};
        const uint8_t* data = bundle.tensor_data(*info);
        return std::string(reinterpret_cast<const char*>(data),
                           reinterpret_cast<const char*>(data + info->nbytes));
    } catch (...) {
        return {};
    }
}

inline double source_audio_seconds_for_voice(const std::string& bundle_path, const std::string& sample_path) {
    if (!sample_path.empty()) {
        const double seconds = wav_duration_seconds_from_file(sample_path);
        if (seconds > 0.0) return seconds;
    }
    return wav_duration_seconds_from_bytes(source_audio_bytes_from_bundle(bundle_path));
}

inline bool run_clone_capture(const ServerConfig& cfg,
                              WorkDispatcher& dispatcher,
                              const std::string& audio_path,
                              const std::string& bundle_path,
                              std::string& error) {
    return dispatcher.submit("clone", audio_path, [&](std::string& task_error, WorkEvent&) {
        bool ok = false;
        std::ostringstream captured;
        std::streambuf* previous_cout = cfg.verbose ? nullptr : std::cout.rdbuf(captured.rdbuf());
        try {
            ok = server_run_tts_clone_native(cfg.model_bundle_dir, audio_path, bundle_path);
        } catch (const std::exception& e) {
            task_error = e.what();
        }
        if (!ok) {
            task_error.clear();
            try {
                ok = server_run_tts_clone_fast(audio_path, bundle_path);
            } catch (const std::exception& e) {
                task_error = e.what();
            }
        }
        if (previous_cout) {
            std::cout.rdbuf(previous_cout);
        }
        if (!ok && task_error.empty()) {
            task_error = "clone failed";
            if (!captured.str().empty()) {
                std::cerr << captured.str();
            }
        }
        return ok;
    }, error);
}

inline bool run_tts_capture(const ServerConfig& cfg,
                            WorkDispatcher& dispatcher,
                            const std::string& voice_bundle,
                            const std::string& input,
                            const std::string& out_wav,
                            std::string& error) {
    return dispatcher.submit("tts", input, [&](std::string& task_error, WorkEvent& event) {
        server_reset_tts_stage_acc();
        bool ok = false;
        std::ostringstream captured;
        std::streambuf* previous_cout = cfg.verbose ? nullptr : std::cout.rdbuf(captured.rdbuf());
        try {
            ok = server_run_tts_product_entry("tts", cfg.model_bundle_dir, voice_bundle, input, out_wav, "standard");
        } catch (const std::exception& e) {
            task_error = e.what();
        }
        if (previous_cout) {
            std::cout.rdbuf(previous_cout);
        }
        if (!ok && task_error.empty()) {
            task_error = "synthesis failed";
            if (!captured.str().empty()) {
                std::cerr << captured.str();
            }
        }
        if (ok) {
            std::ifstream wav(out_wav, std::ios::binary);
            std::string wav_bytes((std::istreambuf_iterator<char>(wav)), std::istreambuf_iterator<char>());
            event.audio_seconds = wav_bytes.size() > 44
                ? static_cast<double>(wav_bytes.size() - 44) / 2.0 / 22050.0
                : 0.0;
        }
        return ok;
    }, error);
}

inline std::optional<VoiceRecord> create_voice_from_audio(VoiceStore& store,
                                                          const ServerConfig& cfg,
                                                          WorkDispatcher& dispatcher,
                                                          const std::string& name,
                                                          const std::string& description,
                                                          const std::string& audio_path,
                                                          bool cleanup_managed_sample,
                                                          std::string& error) {
    const std::string voice_id = make_id("voice");
    const std::string out_bundle = (std::filesystem::path(store.bundles_dir()) / (voice_id + ".pt")).string();
    if (!run_clone_capture(cfg, dispatcher, audio_path, out_bundle, error)) {
        return std::nullopt;
    }
    const std::string now = now_epoch_string();
    const double source_seconds = source_audio_seconds_for_voice(out_bundle, audio_path);
    const bool sample_removed = cleanup_managed_sample &&
        remove_path_under_dir(audio_path, store.samples_dir(), "uploaded voice sample");
    VoiceRecord v{voice_id,
                  name.empty() ? voice_id : name,
                  description,
                  out_bundle,
                  sample_removed ? std::string{} : audio_path,
                  source_seconds,
                  "clone",
                  now,
                  now};
    store.insert_voice(v);
    return v;
}

inline void send_voice_list(int fd, VoiceStore& store) {
    const auto voices = store.list_voices();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < voices.size(); ++i) {
        if (i) out << ",";
        out << voice_json(voices[i]);
    }
    out << "]}";
    send_response(fd, 200, "OK", "application/json", out.str());
}

inline void send_voice_source_audio(int fd, VoiceStore& store, const std::string& id) {
    const auto voice = store.get_voice(id);
    if (!voice) {
        send_json_error(fd, 404, "Not Found", "voice not found");
        return;
    }
    std::string bytes = source_audio_bytes_from_bundle(voice->bundle_path);
    if (bytes.empty() && !voice->sample_path.empty()) {
        bytes = read_file_bytes(voice->sample_path);
    }
    if (bytes.empty()) {
        send_json_error(fd, 404, "Not Found", "source audio not found in voice bundle");
        return;
    }
    send_response(fd, 200, "OK", "audio/wav", bytes);
}

inline void remove_voice_files(VoiceStore& store, const VoiceRecord& voice) {
    bool bundle_still_used = false;
    for (const auto& active : store.list_voices()) {
        if (active.id != voice.id && active.bundle_path == voice.bundle_path) {
            bundle_still_used = true;
            break;
        }
    }
    if (!bundle_still_used) {
        remove_path_under_dir(voice.bundle_path, store.bundles_dir(), "voice bundle");
    }
    remove_path_under_dir(voice.sample_path, store.samples_dir(), "voice sample");
}

inline void send_consent_list(int fd, VoiceStore& store) {
    const auto consents = store.list_consents();
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < consents.size(); ++i) {
        if (i) out << ",";
        out << consent_json(consents[i]);
    }
    out << "]}";
    send_response(fd, 200, "OK", "application/json", out.str());
}

inline bool create_voice_from_request(int fd,
                                      const std::string& head,
                                      const std::string& body,
                                      VoiceStore& store,
                                      const ServerConfig& cfg,
                                      WorkDispatcher& dispatcher) {
    const std::string ct = content_type_from_head(head);
    std::string name;
    std::string description;
    std::string bundle_path;
    std::string sample_path;
    std::string consent_id;
    if (ct.find("multipart/form-data") != std::string::npos) {
        const auto parts = parse_multipart(ct, body);
        name = multipart_value(parts, "name");
        description = multipart_value(parts, "description");
        consent_id = multipart_value(parts, "consent");
        if (auto file = multipart_file(parts, {"audio_sample", "recording", "file", "audio"})) {
            const std::string id = make_id("sample");
            sample_path = (std::filesystem::path(store.samples_dir()) / (id + safe_ext(file->filename, ".wav"))).string();
            save_file_bytes(sample_path, file->data);
        }
    } else {
        name = json_string_field(body, "name");
        description = json_string_field(body, "description");
        bundle_path = json_string_field(body, "bundle_path");
        sample_path = json_string_field(body, "sample_path");
        consent_id = json_string_field(body, "consent");
    }
    if (sample_path.empty() && !consent_id.empty()) {
        if (auto c = store.get_consent(consent_id)) {
            sample_path = c->recording_path;
        }
    }
    if (!bundle_path.empty()) {
        if (store.resolve_voice_bundle(bundle_path).empty()) {
            send_json_error(fd, 400, "Bad Request", "bundle_path is not a usable voice bundle");
            return true;
        }
        const std::string now = now_epoch_string();
        const std::string voice_id = make_id("voice");
        const double source_seconds = source_audio_seconds_for_voice(bundle_path, sample_path);
        VoiceRecord v{voice_id, name.empty() ? voice_id : name, description, bundle_path, sample_path, source_seconds, "import", now, now};
        store.insert_voice(v);
        send_response(fd, 200, "OK", "application/json", voice_json(v));
        return true;
    }
    if (sample_path.empty()) {
        send_json_error(fd, 400, "Bad Request", "missing audio_sample, recording, sample_path, consent, or bundle_path");
        return true;
    }
    std::string error;
    auto v = create_voice_from_audio(store, cfg, dispatcher, name, description, sample_path, true, error);
    if (!v) {
        send_json_error(fd, error == "clone queue is full" ? 429 : 500,
                        error == "clone queue is full" ? "Too Many Requests" : "Internal Server Error",
                        error);
        return true;
    }
    send_response(fd, 200, "OK", "application/json", voice_json(*v));
    return true;
}

inline void handle_speech(int fd,
                          const std::string& body,
                          VoiceStore& store,
                          const ServerConfig& cfg,
                          WorkDispatcher& dispatcher,
                          uint64_t request_id) {
    const auto request_received = std::chrono::steady_clock::now();
    const std::string input = json_string_field(body, "input");
    const std::string voice = json_voice_field(body);
    const std::string response_format = json_string_field(body, "response_format");
    const std::string output_param = json_string_field(body, "output");
    const bool stream_audio = output_param.empty() || json_bool_field(body, "stream", false);
    if (input.empty()) {
        send_json_error(fd, 400, "Bad Request", "missing required field: input");
        return;
    }
    if (!response_format.empty() && response_format != "wav") {
        send_json_error(fd, 400, "Bad Request", "only response_format=wav is supported");
        return;
    }
    const std::string voice_bundle = store.resolve_voice_bundle(voice.empty() ? "sample/qin.pt" : voice);
    if (voice_bundle.empty()) {
        send_json_error(fd, 400, "Bad Request", "voice not found or not a usable voice bundle");
        return;
    }
    std::filesystem::create_directories("outputs");
    const bool keep_output = !output_param.empty();
    const std::string out_wav = keep_output
        ? output_param
        : "outputs/.tmp_req_" + std::to_string(request_id) + ".wav";

    std::cerr << ">> request " << request_id << ": voice=" << voice_bundle
              << " input=" << input.substr(0, 64) << (input.size() > 64 ? "..." : "") << std::endl;

    std::string error;
    if (!run_tts_capture(cfg, dispatcher, voice_bundle, input, out_wav, error)) {
        send_json_error(fd, error == "tts queue is full" ? 429 : 500,
                        error == "tts queue is full" ? "Too Many Requests" : "Internal Server Error",
                        error);
        return;
    }

    std::string wav_bytes;
    {
        std::ifstream wav(out_wav, std::ios::binary);
        wav_bytes.assign(std::istreambuf_iterator<char>(wav), std::istreambuf_iterator<char>());
    }
    if (!keep_output) {
        std::error_code ec;
        const std::filesystem::path out_path(out_wav);
        const std::string stem = out_path.filename().string();
        const std::filesystem::path dir = out_path.parent_path().empty() ? "." : out_path.parent_path();
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            const std::string name = entry.path().filename().string();
            if (name == stem || name.rfind(stem + ".", 0) == 0) {
                std::filesystem::remove_all(entry.path(), ec);
            }
        }
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - request_received).count();
    const double audio_seconds = wav_bytes.size() > 44
        ? static_cast<double>(wav_bytes.size() - 44) / 2.0 / 22050.0
        : 0.0;
    std::cerr << ">> Total request time: " << total << " seconds" << std::endl;
    std::cerr << ">> Generated audio length: " << audio_seconds << " seconds" << std::endl;
    std::cerr << ">> RTF: " << (audio_seconds > 0 ? total / audio_seconds : 0.0) << std::endl;

    if (stream_audio) {
        send_response(fd, 200, "OK", "audio/wav", wav_bytes);
    } else {
        std::ostringstream ack;
        ack << "{\"status\":\"ok\",\"output\":\"" << mtts_json_escape(out_wav)
            << "\",\"audio_seconds\":" << audio_seconds
            << ",\"total_seconds\":" << total
            << ",\"rtf\":" << (audio_seconds > 0 ? total / audio_seconds : 0.0) << "}";
        send_response(fd, 200, "OK", "application/json", ack.str());
    }
}

inline void handle_connection(int fd,
                              VoiceStore& store,
                              const ServerConfig& cfg,
                              WorkDispatcher& dispatcher,
                              uint64_t request_id) {
    std::string head;
    std::string body;
    if (!read_request(fd, head, body)) {
        ::close(fd);
        return;
    }
    std::string method;
    const std::string path = route_path_from_head(head, method);

    try {
        std::string effective_path = path;
        if (path == "/web" || path == "/web/") {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
            } else if (method != "GET") {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            } else {
                send_response(fd, 200, "OK", "text/html; charset=utf-8", web_index_html());
            }
            ::close(fd);
            return;
        }
        if (path == "/web/api/login") {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
            } else if (method != "POST") {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            } else if (!web_key_authorized(cfg, head, body)) {
                send_json_error(fd, 401, "Unauthorized", "invalid web key");
            } else {
                send_response(fd, 200, "OK", "application/json", "{\"ok\":true}");
            }
            ::close(fd);
            return;
        }
        if (path.rfind("/web/api/", 0) == 0) {
            if (!cfg.web_enabled) {
                send_json_error(fd, 404, "Not Found", "web UI is not enabled");
                ::close(fd);
                return;
            }
            if (!web_key_authorized(cfg, head, body)) {
                send_json_error(fd, 401, "Unauthorized", "invalid web key");
                ::close(fd);
                return;
            }
            effective_path = path.substr(std::string("/web/api").size());
            if (effective_path.empty()) {
                effective_path = "/";
            }
        }

        if ((method == "GET") && (effective_path == "/health" || effective_path == "/v1/health")) {
            send_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}");
        } else if (method == "GET" && (effective_path == "/api/status" || effective_path == "/status")) {
            send_response(fd, 200, "OK", "application/json", dispatcher.status_json(cfg));
        } else if (method == "POST" && (effective_path == "/v1/audio/speech" || effective_path == "/speech")) {
            handle_speech(fd, body, store, cfg, dispatcher, request_id);
        } else if (method == "GET" && (effective_path == "/api/voices" || effective_path == "/v1/audio/voices" || effective_path == "/voices")) {
            send_voice_list(fd, store);
        } else if (method == "POST" && (effective_path == "/api/voices" || effective_path == "/v1/audio/voices" || effective_path == "/voices")) {
            create_voice_from_request(fd, head, body, store, cfg, dispatcher);
        } else if (method == "GET" &&
                   (effective_path.rfind("/api/voices/", 0) == 0 ||
                    effective_path.rfind("/v1/audio/voices/", 0) == 0 ||
                    effective_path.rfind("/voices/", 0) == 0) &&
                   effective_path.size() >= std::string("/source-audio").size() &&
                   effective_path.compare(effective_path.size() - std::string("/source-audio").size(),
                                          std::string("/source-audio").size(),
                                          "/source-audio") == 0) {
            const std::string prefix = effective_path.rfind("/api/voices/", 0) == 0
                ? "/api/voices/"
                : (effective_path.rfind("/v1/audio/voices/", 0) == 0 ? "/v1/audio/voices/" : "/voices/");
            std::string id = effective_path.substr(prefix.size());
            id.resize(id.size() - std::string("/source-audio").size());
            send_voice_source_audio(fd, store, id);
        } else if (effective_path.rfind("/api/voices/", 0) == 0 ||
                   effective_path.rfind("/v1/audio/voices/", 0) == 0 ||
                   effective_path.rfind("/voices/", 0) == 0) {
            const std::string prefix = effective_path.rfind("/api/voices/", 0) == 0
                ? "/api/voices/"
                : (effective_path.rfind("/v1/audio/voices/", 0) == 0 ? "/v1/audio/voices/" : "/voices/");
            const std::string id = effective_path.substr(prefix.size());
            if (method == "GET") {
                auto v = store.get_voice(id);
                if (!v) send_json_error(fd, 404, "Not Found", "voice not found");
                else send_response(fd, 200, "OK", "application/json", voice_json(*v));
            } else if (method == "PATCH" || method == "PUT") {
                VoiceRecord patch;
                patch.name = json_string_field(body, "name");
                patch.description = json_string_field(body, "description");
                patch.bundle_path = json_string_field(body, "bundle_path");
                patch.sample_path = json_string_field(body, "sample_path");
                if (!patch.bundle_path.empty() || !patch.sample_path.empty()) {
                    const auto current = store.get_voice(id);
                    const std::string bundle_for_duration = !patch.bundle_path.empty()
                        ? patch.bundle_path
                        : (current ? current->bundle_path : "");
                    const std::string sample_for_duration = !patch.sample_path.empty()
                        ? patch.sample_path
                        : (current ? current->sample_path : "");
                    patch.source_audio_seconds = source_audio_seconds_for_voice(bundle_for_duration, sample_for_duration);
                }
                if (!patch.bundle_path.empty() && store.resolve_voice_bundle(patch.bundle_path).empty()) {
                    send_json_error(fd, 400, "Bad Request", "bundle_path is not a usable voice bundle");
                } else if (!store.update_voice(id, patch)) {
                    send_json_error(fd, 404, "Not Found", "voice not found");
                } else {
                    send_response(fd, 200, "OK", "application/json", voice_json(*store.get_voice(id)));
                }
	            } else if (method == "DELETE") {
	                auto v = store.get_voice(id);
	                if (!v || !store.delete_voice(id)) {
	                    send_json_error(fd, 404, "Not Found", "voice not found");
	                } else {
	                    remove_voice_files(store, *v);
	                    send_response(fd, 200, "OK", "application/json", "{\"deleted\":true}");
	                }
	            } else {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            }
        } else if (method == "GET" && (effective_path == "/v1/audio/voice_consents" || effective_path == "/voice_consents")) {
            send_consent_list(fd, store);
        } else if (method == "POST" && (effective_path == "/v1/audio/voice_consents" || effective_path == "/voice_consents")) {
            const std::string ct = content_type_from_head(head);
            std::string name;
            std::string language;
            std::string recording_path;
            if (ct.find("multipart/form-data") != std::string::npos) {
                const auto parts = parse_multipart(ct, body);
                name = multipart_value(parts, "name");
                language = multipart_value(parts, "language");
                if (auto file = multipart_file(parts, {"recording", "audio_sample", "file", "audio"})) {
                    const std::string sid = make_id("sample");
                    recording_path = (std::filesystem::path(store.samples_dir()) / (sid + safe_ext(file->filename, ".wav"))).string();
                    save_file_bytes(recording_path, file->data);
                }
            } else {
                name = json_string_field(body, "name");
                language = json_string_field(body, "language");
                recording_path = json_string_field(body, "recording_path");
            }
            if (recording_path.empty()) {
                send_json_error(fd, 400, "Bad Request", "missing recording, audio_sample, or recording_path");
            } else {
                const std::string now = now_epoch_string();
                ConsentRecord c{make_id("consent"), name, language, recording_path, now, now};
                store.insert_consent(c);
                std::string voice_id;
                if (json_bool_field(body, "create_voice", true)) {
                    std::string error;
                    auto v = create_voice_from_audio(store, cfg, dispatcher, name, "created from voice consent", recording_path, false, error);
                    if (!v) {
                        send_json_error(fd, error == "clone queue is full" ? 429 : 500,
                                        error == "clone queue is full" ? "Too Many Requests" : "Internal Server Error",
                                        error);
                        ::close(fd);
                        return;
                    }
                    voice_id = v->id;
                }
                send_response(fd, 200, "OK", "application/json", consent_json(c, voice_id));
            }
        } else if (effective_path.rfind("/v1/audio/voice_consents/", 0) == 0 ||
                   effective_path.rfind("/voice_consents/", 0) == 0) {
            const std::string prefix = effective_path.rfind("/v1/audio/voice_consents/", 0) == 0
                ? "/v1/audio/voice_consents/"
                : "/voice_consents/";
            const std::string id = effective_path.substr(prefix.size());
            if (method == "GET") {
                auto c = store.get_consent(id);
                if (!c) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", consent_json(*c));
            } else if (method == "PATCH" || method == "PUT") {
                ConsentRecord patch;
                patch.name = json_string_field(body, "name");
                patch.language = json_string_field(body, "language");
                patch.recording_path = json_string_field(body, "recording_path");
                if (!store.update_consent(id, patch)) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", consent_json(*store.get_consent(id)));
            } else if (method == "DELETE") {
                if (!store.delete_consent(id)) send_json_error(fd, 404, "Not Found", "voice consent not found");
                else send_response(fd, 200, "OK", "application/json", "{\"deleted\":true}");
            } else {
                send_json_error(fd, 405, "Method Not Allowed", "method not allowed");
            }
        } else {
            send_json_error(fd, 404, "Not Found", "unknown endpoint");
        }
    } catch (const std::exception& e) {
        send_json_error(fd, 500, "Internal Server Error", e.what());
    }
    ::close(fd);
}

int run_server(const std::string& host,
               uint16_t port,
               const std::string& model_bundle_dir,
               const std::string& voice_store_dir,
               uint32_t queue_size,
               uint32_t tts_concurrency,
               uint32_t clone_concurrency,
               bool web_enabled,
               const std::string& web_key,
               bool verbose) {
    ::signal(SIGPIPE, SIG_IGN);

    ServerConfig cfg;
    cfg.host = host;
    cfg.port = port;
    cfg.model_bundle_dir = model_bundle_dir;
    cfg.voice_store_dir = voice_store_dir.empty()
        ? env_string("MIT2_VOICE_STORE", "voices")
        : voice_store_dir;
    cfg.queue_size = queue_size == 0 ? env_u32("MIT2_QUEUE_SIZE", 16, 0, 10000) : queue_size;
    cfg.tts_concurrency = tts_concurrency == 0 ? env_u32("MIT2_TTS_CONCURRENCY", 1, 1, 64) : tts_concurrency;
    cfg.clone_concurrency = clone_concurrency == 0 ? env_u32("MIT2_CLONE_CONCURRENCY", 1, 1, 64) : clone_concurrency;
    cfg.web_enabled = web_enabled;
    cfg.web_key = web_key.empty() ? env_string("MIT2_WEBKEY", "") : web_key;
    cfg.verbose = verbose;

    VoiceStore store(cfg.voice_store_dir);
    WorkDispatcher dispatcher(cfg.queue_size);

    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        std::cerr << "error: socket() failed" << std::endl;
        return 1;
    }
    int one = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "error: invalid --host address: " << host << std::endl;
        return 1;
    }
    if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "error: bind " << host << ":" << port << " failed (port in use?)" << std::endl;
        return 1;
    }
    if (::listen(listener, 64) != 0) {
        std::cerr << "error: listen() failed" << std::endl;
        return 1;
    }
    std::filesystem::create_directories("outputs");
    std::cerr << ">> mit2_tts server listening on http://" << host << ":" << port << std::endl;
    std::cerr << ">> endpoints: POST /v1/audio/speech, POST /v1/audio/voice_consents, /api/voices" << std::endl;
    if (cfg.web_enabled) {
        std::cerr << ">> web admin: http://" << host << ":" << port << "/web"
                  << (cfg.web_key.empty() ? "  (no web key configured)" : "") << std::endl;
    }
    std::cerr << ">> model bundle: " << model_bundle_dir << std::endl;
    std::cerr << ">> voice store: " << cfg.voice_store_dir
              << " queue_size=" << cfg.queue_size
              << " tts_concurrency=" << cfg.tts_concurrency
              << " clone_concurrency=" << cfg.clone_concurrency << std::endl;

    std::atomic<uint64_t> request_counter{0};
    std::thread accept_thread([&] {
        for (;;) {
            const int fd = ::accept(listener, nullptr, nullptr);
            if (fd < 0) {
                continue;
            }
            const uint64_t request_id = request_counter.fetch_add(1) + 1;
            std::thread([fd, &store, &cfg, &dispatcher, request_id] {
                handle_connection(fd, store, cfg, dispatcher, request_id);
            }).detach();
        }
    });
    accept_thread.detach();
    dispatcher.run_forever();
    return 0;
}

}  // namespace mit2_server
