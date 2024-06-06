class MCCQConfig():
    dataset_name = 'Multi-modal COVID-CQ'
    is_multi_label = False
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'Favor': 0, 'Against': 1, 'Neutral': 2}
    target_names = ['The use of "Chloroquine" and "Hydroxychloroquine" for the treatment or prevention from the coronavirus.']
    short_target_names = {
        'The use of "Chloroquine" and "Hydroxychloroquine" for the treatment or prevention from the coronavirus.': 'CQ',
    }
    topic_text = {
        'The use of "Chloroquine" and "Hydroxychloroquine" for the treatment or prevention from the coronavirus.': 'The stance on the use of "Chloroquine" and "Hydroxychloroquine" for the treatment or prevention from the coronavirus is:',
    }

    data_dir = 'dataset/Multi-Modal-Stance-Detection/Multi-modal-COVID-CQ'
    image_dir = f'{data_dir}/images'
    in_target_data_dir = f'{data_dir}/in-target'