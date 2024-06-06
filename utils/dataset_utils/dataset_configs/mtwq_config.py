class MTWQConfig():
    dataset_name = 'Multi-modal Taiwan Question'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'Support': 0, 'Oppose': 1, 'Neutral': 2}
    target_names = ['Mainland of China', 'Taiwan of China']
    zero_shot_target_names = ['Mainland of China', 'Taiwan of China']
    short_target_names = {
        'Mainland of China': 'MOC',
        'Taiwan of China': 'TOC',
    }
    topic_text = {
        'Mainland of China': 'The stance on Mainland of China is:',
        'Taiwan of China': 'The stance on Taiwan of China is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Taiwan-Question'
    image_dir = f'{data_dir}/images'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'