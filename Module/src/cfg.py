cfg = {
    "SEED" : 72,
    "train_path" : "/DaiS_internship/Hyundai_RB/save_txt_path/ACC_2000_REJ_274(only_fold_data)/",
    "save_path" : "/DaiS_internship/Hyundai_RB/CHG/Result_0619/",
    "target_size" : [512, 512, 1],
    "weld_type" : True,
    "model_name" : "CNN",
    "epoch" : 500,
    "batch_size" : 16,
    "validation_type" : "cross_validation",
    "fold_num" : 1,
    "learning_rate" : 0.0001,
    "weight_decay" : 0,
    "basic_preprocess" : "basic_normalization",
    "preprocess" : "his_equalized",
    "model_path" : "/DaiS_internship/Hyundai_RB/CJH/single_channel/single_validation/result/CNN_weld/spatial_normalization/h5_model/CNN.h5",
    "margin" : 5
}