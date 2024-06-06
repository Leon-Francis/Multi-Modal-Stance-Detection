class MWTWTConfig():
    dataset_name = "Multi-modal Will-They-Wont-They"
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2, 3]
    label2idx = {'Support': 0, 'Refute': 1, 'Comment': 2, 'Unrelated': 3}
    target_names = [
        'Merger and acquisition between CVS Health and Aetna.',
        'Merger and acquisition between Cigna and Express Scripts.',
        'Merger and acquisition between Anthem and Cigna.', 
        'Merger and acquisition between Aetna and Humana.',
        'Merger and acquisition between Disney and 21st Century Fox.'
    ]
    zero_shot_target_names = [
        'Merger and acquisition between CVS Health and Aetna.',
        'Merger and acquisition between Cigna and Express Scripts.',
        'Merger and acquisition between Anthem and Cigna.',
        'Merger and acquisition between Aetna and Humana.'
    ]
    short_target_names = {
        'Merger and acquisition between CVS Health and Aetna.': 'CSV_AET',
        'Merger and acquisition between Cigna and Express Scripts.': 'CI_ESRX',
        'Merger and acquisition between Anthem and Cigna.': 'ANTM_CI',
        'Merger and acquisition between Aetna and Humana.': 'AET_HUM',
        'Merger and acquisition between Disney and 21st Century Fox.': 'FOXA_DIS'
    }
    topic_text = {
        'Merger and acquisition between CVS Health and Aetna.': 'The stance on merger and acquisition between CVS Health and Aetna is:',
        'Merger and acquisition between Cigna and Express Scripts.': 'The stance on merger and acquisition between Cigna and Express Scripts is:',
        'Merger and acquisition between Anthem and Cigna.': 'The stance on merger and acquisition between Anthem and Cigna is:',
        'Merger and acquisition between Aetna and Humana.': 'The stance on merger and acquisition between Aetna and Humana is:',
        'Merger and acquisition between Disney and 21st Century Fox.': 'The stance on merger and acquisition between Disney and 21st Century Fox is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Will-They-Wont-They'
    image_dir = f'{data_dir}/images'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'