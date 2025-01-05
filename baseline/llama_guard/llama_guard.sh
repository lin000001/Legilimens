export CUDA_VISIBLE_DEVICES=1

# echo hx dataset
# python llama_guard2_query.py --test_dataset hatexplain

# echo mhs dataset
# python llama_guard2_query.py --test_dataset mhs

# echo oig dataset
# python llama_guard2_query.py --test_dataset oig

echo jigsaw dataset
python llama_guard2_query.py --test_dataset jigsaw

# echo beaver dataset
# python llama_guard2_query.py --test_dataset beaver

# echo beaver_adv dataset
# python llama_guard2_query.py --test_dataset beaver_adv

# echo bag dataset
# python llama_guard2_query.py --test_dataset bag

# echo bag_adv dataset
# python llama_guard2_query.py --test_dataset bag_adv