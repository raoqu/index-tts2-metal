// Aggregator for the TTS synthesis implementation.
//
// The implementation is one large translation unit split across several files
// by business area. They are #included in order below; because they share a
// single TU with no forward-declaration header, the include order is
// significant (a function must be defined before a later file uses it).
#include "tts_synthesis/synthesis_pipeline.cpp"
#include "tts_synthesis/product_and_readiness.cpp"
#include "tts_synthesis/clone_manifest_validation.cpp"
#include "tts_synthesis/campplus_speaker_embedding.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers00_02.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers03_05.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers06_09.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers10_13.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers14_15.cpp"
#include "tts_synthesis/w2v_bert_encoder_layers16_17.cpp"
#include "tts_synthesis/voice_clone_and_tests.cpp"
