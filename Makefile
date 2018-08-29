run:
	python run_summarization.py \
		--mode=train \
		--data_path="${HOME}/data/nlp/cnn_dailymail/corenlp/finished_files/val.bin" \
		--vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/word_vocab" \
		--pos_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/pos_vocab" \
		--ner_vocab_path="${HOME}/data/nlp/cnn_dailymail/corenlp/additional/ner_vocab" \
		--log_root=out \
		--exp_name=tmp
