read -p "input training dataset: [mtse, mccq, mwtwt, mruc, mtwq]: " trainDataset
read -p "input train dataset mode: [in_target, zero_shot]: " trainData
read -p "input model framework: [textual, visual, multimodal]: " framework
if [ "$framework" = "textual" ]; then
    read -p "input model name: [bert_base, roberta_base, bertweet_base, robert_base_sentiment, kebert]: " trainModel
fi
if [ "$framework" = "visual" ]; then
    read -p "input model name: [resnet, vit, swin]: " trainModel
fi
if [ "$framework" = "multimodal" ]; then
    read -p "input model name: [bert_vit, roberta_vit, kebert_vit, clip, vilt]: " trainModel
fi
read -p "input running mode: [sweep, wandb, normal]: " runMode
read -p "input training cuda idx: " cudaIdx


currTime=$(date +"%Y-%m-%d_%T")
fileName="run_baseline.py"
outputDir="logs/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${framework}_${trainModel}_${currTime}.log"
nohup python ${fileName} --cuda_idx ${cudaIdx} --dataset_name ${trainDataset} --model_name ${trainModel} --${trainData} --framework_name ${framework} --${runMode} > ${outputName} 2>&1 &