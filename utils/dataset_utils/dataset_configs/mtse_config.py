class MTSEConfig():
    dataset_name = 'Multi-modal Twitter Stance Election 2020'
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'Favor': 0, 'Against': 1, 'Neutral': 2}
    target_names = ['Donald Trump', 'Joe Biden']
    zero_shot_target_names = [
        'Donald Trump',
        'Joe Biden',
    ]
    short_target_names = {
        'Donald Trump': 'DT',
        'Joe Biden': 'JB',
    }
    topic_text = {
        'Donald Trump': 'The stance on Donald Trump is:',
        'Joe Biden': 'The stance on Joe Biden is:'
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-Twitter-Stance-Election-2020'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'