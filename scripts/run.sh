
train_data_dir="../image"
output_dir="../output/"
vocabulary_path="../attr.txt"

# params
learning_rate_attr="1e-2"
learning_rate_obj="1e-3"
num_explanation_tokens_1="50"
num_explanation_tokens_2="50"
word_size="22"
max_train_steps_1="30"
max_train_steps_2="30"

python ../step1.py --learning_rate_attr $learning_rate_attr \
--learning_rate_obj $learning_rate_obj \
--num_explanation_tokens $num_explanation_tokens_1 \
--word_size $word_size \
--max_train_steps $max_train_steps_1\
--train_data_dir $train_data\
--output_dir $output_data\
--vocabulary_path $vocabulary_path\

echo "start step2"
python ../step2.py --learning_rate_attr $learning_rate_attr \
--learning_rate_obj $learning_rate_obj \
--num_explanation_tokens $num_explanation_tokens_2 \
--word_size $word_size \
--max_train_steps $max_train_steps_2\
--train_data_dir $train_data\
--output_dir $output_data\
--saved_params $saved_data\
--vocabulary_path $vocabulary_path\