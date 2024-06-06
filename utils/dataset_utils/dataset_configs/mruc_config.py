class MRUCConfig():
    dataset_name = 'Multi-modal Russo-Ukrainian Conflict'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'Support': 0, 'Oppose': 1, 'Neutral': 2}
    target_names = ['Ukraine', 'Russia']
    zero_shot_target_names = ['Ukraine', 'Russia']
    short_target_names = {
        'Ukraine': 'UKR',
        'Russia': 'RUS',
    }
    topic_text = {
        'Ukraine': 'The stance on Ukraine is:',
        'Russia': 'The stance on Russia is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Russo-Ukrainian-Conflict'
    image_dir = f'{data_dir}/images'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'