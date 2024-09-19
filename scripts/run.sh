
train_data_dir="../image/${category}/${concept}"
output_dir="../output/${category}/${concept}"
dictionary_path="../image/${category}/attr.txt"
output_path="../output/${category}/${concept}/word.txt"

# params
learning_rate_attr="1e-2"
learning_rate_obj="1e-3"
num_explanation_tokens_1="50"
num_explanation_tokens_2="50"
word_size="22"
max_train_steps_1="30"
max_train_steps_2="30"

i=0
while true; do
    train_data="${train_data_dir}/${i}"

    # 检查文件夹是否存在
    if [ -d "${train_data}" ]; then
        output_data="${output_dir}/${i}"
        saved_data="${output_data}/step1_params.pt"

        echo "start step1"

        python ../step1.py --learning_rate_attr $learning_rate_attr \
        --learning_rate_obj $learning_rate_obj \
        --num_explanation_tokens $num_explanation_tokens_1 \
        --word_size $word_size \
        --max_train_steps $max_train_steps_1\
        --train_data_dir $train_data\
        --output_dir $output_data\
        --dictionary_path $dictionary_path\

        echo "start step2"
        python ../step2.py --learning_rate_attr $learning_rate_attr \
        --learning_rate_obj $learning_rate_obj \
        --num_explanation_tokens $num_explanation_tokens_2 \
        --word_size $word_size \
        --max_train_steps $max_train_steps_2\
        --train_data_dir $train_data\
        --output_dir $output_data\
        --saved_params $saved_data\
        --dictionary_path $dictionary_path\
        --output_path $output_path

        i=$((i + 1))
        echo "i: ${i}"

    else
        echo "folder ${train_data} not exists"
        break
    fi
done