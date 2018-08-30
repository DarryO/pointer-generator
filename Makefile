run128:
	python run_summarization.py \
		--mode=train \
		--data_path="${HOME}/data/nlp/cnn_dailymail/corenlp/finished_files/val.bin" \
		--vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/word_vocab" \
		--pos_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/pos_vocab" \
		--ner_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/ner_vocab" \
		--sem_emb_dim=128 \
		--log_root=out \
		--exp_name=sem_128
run50:
	python run_summarization.py \
		--mode=train \
		--data_path="${HOME}/data/nlp/cnn_dailymail/corenlp/finished_files/val.bin" \
		--vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/word_vocab" \
		--pos_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/pos_vocab" \
		--ner_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/ner_vocab" \
		--sem_emb_dim=50 \
		--log_root=out \
		--exp_name=sem_50
