read -p "input training dataset: [mtse, mccq, mwtwt, mruc, mtwq]: " trainDataset
read -p "input train dataset mode: [in_target, zero_shot]: " trainData
read -p "input model framework: [tmpt, tmpt_gpt_cot]: " framework
read -p "input model name: [bert_vit, roberta_vit, kebert_vit]: " trainModel
read -p "input running mode: [sweep, wandb, normal]: " runMode
read -p "input training cuda idx: " cudaIdx

currTime=$(date +"%Y-%m-%d_%T")
fileName="run_tmpt.py"
outputDir="tmpt_logs/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${framework}_${trainModel}_${currTime}.log"
nohup python ${fileName} --cuda_idx ${cudaIdx} --dataset_name ${trainDataset} --model_name ${trainModel} --${trainData} --framework_name ${framework} --${runMode} > ${outputName} 2>&1 &