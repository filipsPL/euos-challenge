#!/usr/bin/env python3

estimator_list = [
    "catboost",
    "rf",
    "xgboost",
    "extra_tree",
    "lgbm",
    "xgb_limitdepth",
    "xgboost",
]


# warm_start_configs = { }

# python extract_best_configs_loop_better.py  ./5-predictions/5-flaml-2D+RdkitFP+dyesSelected+QM9pred+easyTargetPreds+xtb3_noWeights+warmStart

## Clusters
'''

warm_start_configs = {
    # Top 5 configurations for catboost
    'catboost': [
        {"early_stopping_rounds":14,"learning_rate":0.005349303918299971,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 1: metric=0.110407
        {"early_stopping_rounds":18,"learning_rate":0.008296265166801306,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 2: metric=0.110420
        {"early_stopping_rounds":18,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 3: metric=0.110600
        {"early_stopping_rounds":20,"learning_rate":0.006989682262157448,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 4: metric=0.110618
        {"early_stopping_rounds":11,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 5: metric=0.110624
        {"early_stopping_rounds":14,"learning_rate":0.038196035074171276,"n_estimators":8192,"FLAML_sample_size":69943},  # Rank 1: metric=0.378274
        {"early_stopping_rounds":13,"learning_rate":0.11131853910196476,"n_estimators":8192,"FLAML_sample_size":69943},  # Rank 2: metric=0.379786
        {"early_stopping_rounds":15,"learning_rate":0.10826171139791693,"n_estimators":8192,"FLAML_sample_size":69943},  # Rank 3: metric=0.382008
        {"early_stopping_rounds":10,"learning_rate":0.035621648298210684,"n_estimators":8192,"FLAML_sample_size":69943},  # Rank 4: metric=0.393471
        {"early_stopping_rounds":17,"learning_rate":0.04967427143132433,"n_estimators":8192,"FLAML_sample_size":69943},  # Rank 5: metric=0.397576
        {"early_stopping_rounds":15,"learning_rate":0.008841908434960067,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 1: metric=0.135979
        {"early_stopping_rounds":15,"learning_rate":0.01438865234821595,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 2: metric=0.136100
        {"early_stopping_rounds":15,"learning_rate":0.0051180198454968745,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 3: metric=0.136146
        {"early_stopping_rounds":16,"learning_rate":0.007174935939201643,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 4: metric=0.136172
        {"early_stopping_rounds":14,"learning_rate":0.011517986011034723,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 5: metric=0.136270
        {"early_stopping_rounds":19,"learning_rate":0.011157106854298368,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 1: metric=0.314773
        {"early_stopping_rounds":20,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 2: metric=0.315031
        {"early_stopping_rounds":17,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 3: metric=0.315031
        {"early_stopping_rounds":18,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 4: metric=0.315031
        {"early_stopping_rounds":16,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":69942},  # Rank 5: metric=0.315379
    ],

    # Top 5 configurations for extra_tree
    'extra_tree': [
        {"n_estimators":1119,"max_features":0.6460704819803129,"max_leaves":7487,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 1: metric=0.108129
        {"n_estimators":1282,"max_features":0.6075531594664093,"max_leaves":10400,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 2: metric=0.108297
        {"n_estimators":1040,"max_features":0.5394826058192508,"max_leaves":6921,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 3: metric=0.108304
        {"n_estimators":752,"max_features":0.6891336674999728,"max_leaves":5981,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 4: metric=0.108411
        {"n_estimators":904,"max_features":0.6208979401237738,"max_leaves":15469,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 5: metric=0.108635
        {"n_estimators":297,"max_features":0.10198557128593509,"max_leaves":216,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 1: metric=0.371905
        {"n_estimators":149,"max_features":0.654724126188905,"max_leaves":9820,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 2: metric=0.373038
        {"n_estimators":1813,"max_features":0.10989472075192255,"max_leaves":101,"criterion":'gini',"FLAML_sample_size":49658},  # Rank 3: metric=0.375762
        {"n_estimators":206,"max_features":0.43974301596542176,"max_leaves":689,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 4: metric=0.376517
        {"n_estimators":795,"max_features":0.08388864914293526,"max_leaves":100,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 5: metric=0.389608
        {"n_estimators":845,"max_features":1.0,"max_leaves":618,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 1: metric=0.126890
        {"n_estimators":698,"max_features":1.0,"max_leaves":717,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 2: metric=0.126966
        {"n_estimators":982,"max_features":0.9549386336830441,"max_leaves":812,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 3: metric=0.127010
        {"n_estimators":1130,"max_features":0.797546994264138,"max_leaves":671,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 4: metric=0.127251
        {"n_estimators":809,"max_features":1.0,"max_leaves":479,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 5: metric=0.127350
        {"n_estimators":1082,"max_features":0.28983580021492084,"max_leaves":77,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 1: metric=0.310687
        {"n_estimators":2047,"max_features":0.23919424420166407,"max_leaves":76,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 2: metric=0.311957
        {"n_estimators":2047,"max_features":0.39778425050192834,"max_leaves":35,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 3: metric=0.312233
        {"n_estimators":2047,"max_features":0.31402973608471574,"max_leaves":127,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 4: metric=0.313153
        {"n_estimators":2047,"max_features":0.16114692024644578,"max_leaves":99,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 5: metric=0.314961

    ],

    # Top 5 configurations for lgbm
    'lgbm': [
        {"n_estimators":512,"num_leaves":91,"min_child_samples":11,"learning_rate":0.0025342048092720382,"log_max_bin":9,"colsample_bytree":0.8637461166623264,"reg_alpha":0.15484182403782193,"reg_lambda":0.09248110575789187,"FLAML_sample_size":49658},  # Rank 1: metric=0.111396
        {"n_estimators":477,"num_leaves":56,"min_child_samples":7,"learning_rate":0.005845178078086581,"log_max_bin":7,"colsample_bytree":0.8084829150550453,"reg_alpha":1.0285186200871819,"reg_lambda":0.3495503291116563,"FLAML_sample_size":49658},  # Rank 2: metric=0.111506
        {"n_estimators":462,"num_leaves":104,"min_child_samples":10,"learning_rate":0.0065053138195699375,"log_max_bin":7,"colsample_bytree":0.9773147444781976,"reg_alpha":0.07157349205872487,"reg_lambda":0.525667212673718,"FLAML_sample_size":49658},  # Rank 3: metric=0.111844
        {"n_estimators":433,"num_leaves":36,"min_child_samples":11,"learning_rate":0.0039139989180129505,"log_max_bin":7,"colsample_bytree":0.867972329424523,"reg_alpha":0.9396588828692268,"reg_lambda":0.11996935042037567,"FLAML_sample_size":49658},  # Rank 4: metric=0.111874
        {"n_estimators":1926,"num_leaves":57,"min_child_samples":9,"learning_rate":0.0036595801911476203,"log_max_bin":7,"colsample_bytree":0.7643741630453937,"reg_alpha":0.343098500020875,"reg_lambda":0.07566748754876663,"FLAML_sample_size":69943},  # Rank 5: metric=0.112057
        {"n_estimators":5,"num_leaves":64,"min_child_samples":125,"learning_rate":0.026822299108131054,"log_max_bin":5,"colsample_bytree":0.23304472552149,"reg_alpha":0.5127929178813916,"reg_lambda":0.06904243553895845,"FLAML_sample_size":49658},  # Rank 1: metric=0.355224
        {"n_estimators":145,"num_leaves":256,"min_child_samples":9,"learning_rate":0.003132140623239833,"log_max_bin":6,"colsample_bytree":0.8994125388185525,"reg_alpha":0.009357561401367589,"reg_lambda":1.0626932216724168,"FLAML_sample_size":49658},  # Rank 2: metric=0.359927
        {"n_estimators":11,"num_leaves":128,"min_child_samples":114,"learning_rate":0.02517230877639018,"log_max_bin":6,"colsample_bytree":0.38996713838637465,"reg_alpha":5.4447913689079765,"reg_lambda":0.06384454446768241,"FLAML_sample_size":69943},  # Rank 3: metric=0.377445
        {"n_estimators":10,"num_leaves":29,"min_child_samples":53,"learning_rate":0.04276697562214586,"log_max_bin":5,"colsample_bytree":0.3566115562519952,"reg_alpha":0.5711083754648654,"reg_lambda":0.03407222984607694,"FLAML_sample_size":69943},  # Rank 4: metric=0.380950
        {"n_estimators":4,"num_leaves":159,"min_child_samples":128,"learning_rate":0.04044877824124652,"log_max_bin":4,"colsample_bytree":0.2783468384057332,"reg_alpha":0.2385734936453874,"reg_lambda":0.0031491986288832473,"FLAML_sample_size":69943},  # Rank 5: metric=0.386435
        {"n_estimators":878,"num_leaves":316,"min_child_samples":4,"learning_rate":0.0033700699847521412,"log_max_bin":6,"colsample_bytree":0.8478876113343088,"reg_alpha":0.37765650632694386,"reg_lambda":0.04524306326267369,"FLAML_sample_size":69942},  # Rank 1: metric=0.129393
        {"n_estimators":369,"num_leaves":139,"min_child_samples":9,"learning_rate":0.008814692242002733,"log_max_bin":6,"colsample_bytree":0.8537850279392886,"reg_alpha":0.47851144568710213,"reg_lambda":0.01594386710971353,"FLAML_sample_size":69942},  # Rank 2: metric=0.130022
        {"n_estimators":421,"num_leaves":127,"min_child_samples":12,"learning_rate":0.00985373037297579,"log_max_bin":7,"colsample_bytree":0.7091399422239761,"reg_alpha":4.356911740874267,"reg_lambda":0.007414007024647386,"FLAML_sample_size":69942},  # Rank 3: metric=0.131163
        {"n_estimators":324,"num_leaves":152,"min_child_samples":5,"learning_rate":0.00788521670273371,"log_max_bin":5,"colsample_bytree":0.998430113654601,"reg_alpha":0.05255401469473293,"reg_lambda":0.03428738299911374,"FLAML_sample_size":69942},  # Rank 4: metric=0.132147
        {"n_estimators":1434,"num_leaves":147,"min_child_samples":16,"learning_rate":0.0051020246475882245,"log_max_bin":6,"colsample_bytree":0.8143413744196009,"reg_alpha":3.1313944284061686,"reg_lambda":0.024449785513150718,"FLAML_sample_size":69942},  # Rank 5: metric=0.132231
        {"n_estimators":1284,"num_leaves":40,"min_child_samples":27,"learning_rate":0.0028739471959962396,"log_max_bin":6,"colsample_bytree":0.7767866726606087,"reg_alpha":0.23205575154427052,"reg_lambda":1.7533873161798996,"FLAML_sample_size":69942},  # Rank 1: metric=0.296656
        {"n_estimators":472,"num_leaves":31,"min_child_samples":21,"learning_rate":0.0058625553385260225,"log_max_bin":7,"colsample_bytree":0.6850943442618156,"reg_alpha":1.5831652614415697,"reg_lambda":0.6600631771728843,"FLAML_sample_size":69942},  # Rank 2: metric=0.298292
        {"n_estimators":997,"num_leaves":49,"min_child_samples":58,"learning_rate":0.002655832477749344,"log_max_bin":7,"colsample_bytree":0.7765504178235021,"reg_alpha":0.026053918258646596,"reg_lambda":0.8772113930042703,"FLAML_sample_size":69942},  # Rank 3: metric=0.299099
        {"n_estimators":1123,"num_leaves":70,"min_child_samples":9,"learning_rate":0.0022413966634218024,"log_max_bin":7,"colsample_bytree":0.6791969276568358,"reg_alpha":1.2494845566665234,"reg_lambda":1.8730261533602792,"FLAML_sample_size":69942},  # Rank 4: metric=0.299760
        {"n_estimators":2839,"num_leaves":15,"min_child_samples":11,"learning_rate":0.002011307621853207,"log_max_bin":7,"colsample_bytree":0.8458449562661887,"reg_alpha":2.9779738203950448,"reg_lambda":0.4623421495156771,"FLAML_sample_size":69942},  # Rank 5: metric=0.300822

    ],

    # Top 5 configurations for rf
    'rf': [
        {"n_estimators":2010,"max_features":0.278700005692999,"max_leaves":602,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 1: metric=0.109971
        {"n_estimators":1186,"max_features":0.21229997279838367,"max_leaves":365,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 2: metric=0.110175
        {"n_estimators":911,"max_features":0.15005629995863315,"max_leaves":1360,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 3: metric=0.110405
        {"n_estimators":1520,"max_features":0.5141239741732394,"max_leaves":1090,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 4: metric=0.110417
        {"n_estimators":1457,"max_features":0.13040969249846804,"max_leaves":1992,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 5: metric=0.110466
        {"n_estimators":911,"max_features":0.15005629995863318,"max_leaves":1360,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 1: metric=0.364766
        {"n_estimators":2010,"max_features":0.27870000569299896,"max_leaves":602,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 2: metric=0.372727
        {"n_estimators":2047,"max_features":0.4634172080216634,"max_leaves":285,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 3: metric=0.385275
        {"n_estimators":505,"max_features":0.3049954655109954,"max_leaves":246,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 4: metric=0.389805
        {"n_estimators":986,"max_features":0.11565832274882688,"max_leaves":22795,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 5: metric=0.398851
        {"n_estimators":1625,"max_features":0.33744286677471147,"max_leaves":504,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 1: metric=0.127805
        {"n_estimators":2047,"max_features":0.6736925090533447,"max_leaves":750,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 2: metric=0.127806
        {"n_estimators":1198,"max_features":1.0,"max_leaves":577,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 3: metric=0.128027
        {"n_estimators":1920,"max_features":0.3137126979172228,"max_leaves":2923,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 4: metric=0.128947
        {"n_estimators":576,"max_features":0.4610968980560432,"max_leaves":513,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 5: metric=0.129000
        {"n_estimators":1769,"max_features":0.2402695815540889,"max_leaves":59,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 1: metric=0.303105
        {"n_estimators":2047,"max_features":0.37204050625001756,"max_leaves":76,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 2: metric=0.303253
        {"n_estimators":2047,"max_features":0.6187099293670695,"max_leaves":35,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 3: metric=0.305981
        {"n_estimators":872,"max_features":0.5345368099953893,"max_leaves":50,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 4: metric=0.306527
        {"n_estimators":1208,"max_features":0.27010923835459744,"max_leaves":33,"criterion":'entropy',"FLAML_sample_size":69942},  # Rank 5: metric=0.307190
        ],

    # Top 5 configurations for xgb_limitdepth
    'xgb_limitdepth': [
        {"n_estimators":735,"max_depth":6,"min_child_weight":0.5113636175923822,"learning_rate":0.0056209987061448755,"subsample":0.802253968365345,"colsample_bylevel":0.40769341368611844,"colsample_bytree":0.8935202301073101,"reg_alpha":0.22267890085926173,"reg_lambda":59.49734012706947,"FLAML_sample_size":49658},  # Rank 1: metric=0.110856
        {"n_estimators":576,"max_depth":7,"min_child_weight":0.8172369729779779,"learning_rate":0.008306655675059052,"subsample":0.7209090713757365,"colsample_bylevel":0.4507618530951358,"colsample_bytree":0.8415694158800056,"reg_alpha":0.0013766853389870423,"reg_lambda":9.410946760391326,"FLAML_sample_size":49658},  # Rank 2: metric=0.110969
        {"n_estimators":210,"max_depth":6,"min_child_weight":0.8318198173275783,"learning_rate":0.007371658676919793,"subsample":0.9359307399050656,"colsample_bylevel":0.40843666999901823,"colsample_bytree":0.8473431201117178,"reg_alpha":0.00511752026706661,"reg_lambda":8.016682702428735,"FLAML_sample_size":49658},  # Rank 3: metric=0.111003
        {"n_estimators":383,"max_depth":7,"min_child_weight":1.420929211420044,"learning_rate":0.0025148506288444586,"subsample":0.6407578925147996,"colsample_bylevel":0.5011723104767596,"colsample_bytree":0.82885576759612,"reg_alpha":0.021998672690898093,"reg_lambda":3.8814294819523854,"FLAML_sample_size":49658},  # Rank 4: metric=0.111074
        {"n_estimators":576,"max_depth":7,"min_child_weight":0.8172369729779779,"learning_rate":0.008306655675059052,"subsample":0.7209090713757365,"colsample_bylevel":0.45076185309513583,"colsample_bytree":0.8415694158800056,"reg_alpha":0.01182710488316438,"reg_lambda":9.410946760391326,"FLAML_sample_size":49658},  # Rank 5: metric=0.111101
        {"n_estimators":383,"max_depth":7,"min_child_weight":1.420929211420044,"learning_rate":0.0025148506288444586,"subsample":0.6407578925147996,"colsample_bylevel":0.5011723104767596,"colsample_bytree":0.82885576759612,"reg_alpha":0.021998672690898093,"reg_lambda":3.8814294819523854,"FLAML_sample_size":49658},  # Rank 1: metric=0.346323
        {"n_estimators":275,"max_depth":7,"min_child_weight":3.079963574401478,"learning_rate":0.006055398634218055,"subsample":0.5014395554459854,"colsample_bylevel":0.6156264097786738,"colsample_bytree":0.8910953960721342,"reg_alpha":0.11235878919896915,"reg_lambda":0.7967924764668229,"FLAML_sample_size":69943},  # Rank 2: metric=0.366677
        {"n_estimators":88,"max_depth":7,"min_child_weight":4.0226033803073555,"learning_rate":0.004994733259560688,"subsample":0.6069709493711013,"colsample_bylevel":0.44378655429034874,"colsample_bytree":0.9741097693561845,"reg_alpha":0.11956298168467534,"reg_lambda":1.5146734481822137,"FLAML_sample_size":69943},  # Rank 3: metric=0.368822
        {"n_estimators":2567,"max_depth":7,"min_child_weight":7.4273842216318435,"learning_rate":0.004459977413947981,"subsample":0.5577532707820331,"colsample_bylevel":0.5287166712533805,"colsample_bytree":0.8736047240872873,"reg_alpha":0.039888525497280815,"reg_lambda":10.970652797339108,"FLAML_sample_size":69943},  # Rank 4: metric=0.368837
        {"n_estimators":157,"max_depth":7,"min_child_weight":6.9280624860053095,"learning_rate":0.0015799438755677154,"subsample":0.7083397911186939,"colsample_bylevel":0.33136349973807655,"colsample_bytree":0.7733674804613341,"reg_alpha":0.006228240556357768,"reg_lambda":16.545224219584075,"FLAML_sample_size":69943},  # Rank 5: metric=0.370500
        {"n_estimators":576,"max_depth":7,"min_child_weight":0.8172369729779779,"learning_rate":0.008306655675059052,"subsample":0.7209090713757365,"colsample_bylevel":0.4507618530951358,"colsample_bytree":0.8415694158800056,"reg_alpha":0.0013766853389870423,"reg_lambda":9.410946760391326,"FLAML_sample_size":49658},  # Rank 1: metric=0.132365
        {"n_estimators":383,"max_depth":7,"min_child_weight":1.420929211420044,"learning_rate":0.0025148506288444586,"subsample":0.6407578925147996,"colsample_bylevel":0.5011723104767596,"colsample_bytree":0.82885576759612,"reg_alpha":0.021998672690898093,"reg_lambda":3.8814294819523854,"FLAML_sample_size":49658},  # Rank 2: metric=0.132618
        {"n_estimators":576,"max_depth":7,"min_child_weight":0.8172369729779779,"learning_rate":0.008306655675059052,"subsample":0.7209090713757365,"colsample_bylevel":0.45076185309513583,"colsample_bytree":0.8415694158800056,"reg_alpha":0.01182710488316438,"reg_lambda":9.410946760391326,"FLAML_sample_size":49658},  # Rank 3: metric=0.132364
        {"n_estimators":210,"max_depth":6,"min_child_weight":0.8318198173275783,"learning_rate":0.007371658676919793,"subsample":0.9359307399050656,"colsample_bylevel":0.40843666999901823,"colsample_bytree":0.8473431201117178,"reg_alpha":0.00511752026706661,"reg_lambda":8.016682702428735,"FLAML_sample_size":49658},  # Rank 4: metric=0.132743
        {"n_estimators":1989,"max_depth":5,"min_child_weight":0.055171411410764676,"learning_rate":0.0017656745030465285,"subsample":1.0,"colsample_bylevel":0.16131114744086472,"colsample_bytree":0.9929951727638029,"reg_alpha":1.4246358171700497,"reg_lambda":3.275174244768462,"FLAML_sample_size":69942},  # Rank 1: metric=0.297269
        {"n_estimators":303,"max_depth":7,"min_child_weight":0.06315095605871969,"learning_rate":0.004402002471202711,"subsample":0.869961090968031,"colsample_bylevel":0.15078842864918476,"colsample_bytree":1.0,"reg_alpha":0.5600476746188936,"reg_lambda":0.2823144381611158,"FLAML_sample_size":69942},  # Rank 2: metric=0.297863
        {"n_estimators":1555,"max_depth":8,"min_child_weight":0.3197256733631024,"learning_rate":0.0011243149453923206,"subsample":1.0,"colsample_bylevel":0.12568681585781138,"colsample_bytree":1.0,"reg_alpha":0.08888224316066647,"reg_lambda":0.021176127648350113,"FLAML_sample_size":69942},  # Rank 3: metric=0.297979
        {"n_estimators":4435,"max_depth":7,"min_child_weight":0.07597141518927315,"learning_rate":0.0009765625,"subsample":0.9702222201670097,"colsample_bylevel":0.13402258418859578,"colsample_bytree":1.0,"reg_alpha":3.253242463144427,"reg_lambda":0.556986498965531,"FLAML_sample_size":69942},  # Rank 4: metric=0.299005
        {"n_estimators":666,"max_depth":9,"min_child_weight":0.6486888214976027,"learning_rate":0.005120166119273544,"subsample":0.8781827714199612,"colsample_bylevel":0.4400521954839083,"colsample_bytree":1.0,"reg_alpha":0.38232331627049615,"reg_lambda":25.39044237207955,"FLAML_sample_size":69942},  # Rank 5: metric=0.299822
    ],

    # Top 5 configurations for xgboost
    'xgboost': [
        {"n_estimators":640,"max_leaves":4,"min_child_weight":2.791713219362991,"learning_rate":0.00344139232687018,"subsample":0.7667775257572661,"colsample_bylevel":0.7833858742981361,"colsample_bytree":0.9060222225840808,"reg_alpha":3.159682870586172,"reg_lambda":5.540753876300501,"FLAML_sample_size":10000},  # Rank 1: metric=0.109833
        {"n_estimators":1758,"max_leaves":9,"min_child_weight":2.7427709863232814,"learning_rate":0.003877887237454888,"subsample":0.5517558572279371,"colsample_bylevel":0.8257110573942535,"colsample_bytree":0.9002485183523686,"reg_alpha":0.8499993857919501,"reg_lambda":6.5044035890928225,"FLAML_sample_size":10000},  # Rank 2: metric=0.109911
        {"n_estimators":793,"max_leaves":4,"min_child_weight":0.20930608709478724,"learning_rate":0.010531403803914237,"subsample":0.470720577481042,"colsample_bylevel":0.8629207773752521,"colsample_bytree":0.8950924257570008,"reg_alpha":0.1368424625993546,"reg_lambda":181.63184380444,"FLAML_sample_size":10000},  # Rank 3: metric=0.110499
        {"n_estimators":697,"max_leaves":18,"min_child_weight":0.35692692493635836,"learning_rate":0.0121112783899978,"subsample":0.47978440075808826,"colsample_bylevel":1.0,"colsample_bytree":0.6441209675031997,"reg_alpha":0.0009765625,"reg_lambda":34.10603888022801,"FLAML_sample_size":10000},  # Rank 4: metric=0.111284
        {"n_estimators":262,"max_leaves":46,"min_child_weight":0.16641319268820595,"learning_rate":0.020457898238124002,"subsample":0.7294081695369045,"colsample_bylevel":0.7846555696337353,"colsample_bytree":0.7293484552152706,"reg_alpha":0.0047947968365186725,"reg_lambda":8.418017100025061,"FLAML_sample_size":10000},  # Rank 5: metric=0.112524
        {"n_estimators":594,"max_leaves":4,"min_child_weight":3.749023161601346,"learning_rate":0.022761063063756377,"subsample":0.17949957317579163,"colsample_bylevel":0.7296094171702864,"colsample_bytree":0.3438431691944703,"reg_alpha":0.19982822088388388,"reg_lambda":2.466965361991524,"FLAML_sample_size":69943},  # Rank 1: metric=0.343738
        {"n_estimators":721,"max_leaves":12,"min_child_weight":16.005676207841205,"learning_rate":0.0395590360872144,"subsample":0.29455124508108377,"colsample_bylevel":0.8673183271585562,"colsample_bytree":0.4056261511080759,"reg_alpha":0.3971510419552766,"reg_lambda":0.5616662317721264,"FLAML_sample_size":69943},  # Rank 2: metric=0.345814
        {"n_estimators":171,"max_leaves":4,"min_child_weight":2.9344346842728877,"learning_rate":0.03821799465974591,"subsample":0.1363314699146491,"colsample_bylevel":0.8002784087696613,"colsample_bytree":0.2577393236288503,"reg_alpha":2.862579747429082,"reg_lambda":2.2913459696533267,"FLAML_sample_size":69943},  # Rank 3: metric=0.347506
        {"n_estimators":129,"max_leaves":21,"min_child_weight":4.469720668225134,"learning_rate":0.012241129861026153,"subsample":0.1454121109753269,"colsample_bylevel":0.8820682108322971,"colsample_bytree":0.6383615407062038,"reg_alpha":0.6387017942333303,"reg_lambda":13.352732270597246,"FLAML_sample_size":69943},  # Rank 4: metric=0.358562
        {"n_estimators":3137,"max_leaves":4,"min_child_weight":3.8120316177802676,"learning_rate":0.017177395152952614,"subsample":0.16542563699871712,"colsample_bylevel":0.6603152681117364,"colsample_bytree":0.4768630461587562,"reg_alpha":1.3992764576739112,"reg_lambda":10.193892930076203,"FLAML_sample_size":69943},  # Rank 5: metric=0.361318
        {"n_estimators":137,"max_leaves":116,"min_child_weight":10.531506842041,"learning_rate":0.012119538751782141,"subsample":0.7619523955331455,"colsample_bylevel":1.0,"colsample_bytree":0.8009793559735034,"reg_alpha":0.0009765625,"reg_lambda":2.0090857357006815,"FLAML_sample_size":69942},  # Rank 1: metric=0.131106
        {"n_estimators":122,"max_leaves":52,"min_child_weight":9.869865705539803,"learning_rate":0.01557381793041858,"subsample":0.6715436752664907,"colsample_bylevel":1.0,"colsample_bytree":1.0,"reg_alpha":0.002889055776384908,"reg_lambda":1.6923685453856048,"FLAML_sample_size":69942},  # Rank 2: metric=0.131116
        {"n_estimators":128,"max_leaves":191,"min_child_weight":6.118558323554557,"learning_rate":0.020401520016661664,"subsample":0.795214926034737,"colsample_bylevel":1.0,"colsample_bytree":0.9227606819338154,"reg_alpha":0.0009765625,"reg_lambda":3.8272941125844566,"FLAML_sample_size":69942},  # Rank 3: metric=0.131434
        {"n_estimators":46,"max_leaves":45,"min_child_weight":10.019530735652443,"learning_rate":0.015202570473986023,"subsample":0.6732050435388284,"colsample_bylevel":0.9445232195703459,"colsample_bytree":0.8409490279701145,"reg_alpha":0.014184893949067627,"reg_lambda":5.772983074431339,"FLAML_sample_size":69942},  # Rank 4: metric=0.132655
        {"n_estimators":337,"max_leaves":17,"min_child_weight":9.725119650390443,"learning_rate":0.007511752465239822,"subsample":0.6643921076123083,"colsample_bylevel":0.9210923084620792,"colsample_bytree":0.9281757775452156,"reg_alpha":0.05918813042359323,"reg_lambda":7.329895669999624,"FLAML_sample_size":69942},  # Rank 5: metric=0.133148
        {"n_estimators":147,"max_leaves":19,"min_child_weight":0.028074467067560165,"learning_rate":0.011221608923919093,"subsample":0.5411922764812254,"colsample_bylevel":0.7229058327698051,"colsample_bytree":0.7132132772599553,"reg_alpha":0.01847766834156327,"reg_lambda":14.989604984506244,"FLAML_sample_size":69942},  # Rank 1: metric=0.307043
        {"n_estimators":3676,"max_leaves":12,"min_child_weight":0.0260048152852781,"learning_rate":0.004134360299654334,"subsample":0.5398574644765065,"colsample_bylevel":0.7788012711926892,"colsample_bytree":0.9993271961638156,"reg_alpha":0.3839323200339809,"reg_lambda":55.917532245975636,"FLAML_sample_size":69942},  # Rank 2: metric=0.308568
        {"n_estimators":154,"max_leaves":70,"min_child_weight":0.017404012301724,"learning_rate":0.014700176931780232,"subsample":0.6648635272494716,"colsample_bylevel":0.7934147544509016,"colsample_bytree":0.6359739591937708,"reg_alpha":0.001525698653057009,"reg_lambda":33.89901511913037,"FLAML_sample_size":69942},  # Rank 3: metric=0.309565
        {"n_estimators":640,"max_leaves":22,"min_child_weight":0.09615684327051113,"learning_rate":0.009843749848106487,"subsample":0.4662489739024849,"colsample_bylevel":0.8185995243155134,"colsample_bytree":1.0,"reg_alpha":1.3421142873965928,"reg_lambda":125.0845589121639,"FLAML_sample_size":69942},  # Rank 4: metric=0.309742
        {"n_estimators":2390,"max_leaves":9,"min_child_weight":0.1700781134781664,"learning_rate":0.004617864624307387,"subsample":0.38775302484108876,"colsample_bylevel":0.7468352023286624,"colsample_bytree":0.9442324764966012,"reg_alpha":0.14233787897358546,"reg_lambda":225.43687951059073,"FLAML_sample_size":69942},  # Rank 5: metric=0.310794
        ],
}
'''

# Warm start configurations for FLAML
# Representative configurations (hybrid selection: top 20.0% + K-Means + best per cluster)
# Auto-generated from optimization logs
# specyficznie dla fluor 480+

warm_start_configs = {
    # Top 5 configurations for catboost
    'catboost': [
        {"early_stopping_rounds":18,"learning_rate":0.008296265166801306,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 1: metric=0.493005
        {"early_stopping_rounds":14,"learning_rate":0.005349303918299971,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 2: metric=0.494314
        {"early_stopping_rounds":20,"learning_rate":0.006989682262157448,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 3: metric=0.498520
        {"early_stopping_rounds":11,"learning_rate":0.005,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 4: metric=0.498733
        {"early_stopping_rounds":14,"learning_rate":0.038196035074171276,"n_estimators":8192,"FLAML_sample_size":10000},  # Rank 5: metric=0.503128
    ],

    # Top 5 configurations for extra_tree
    'extra_tree': [
        {"n_estimators":2047,"max_features":0.39778425050192817,"max_leaves":35,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 1: metric=0.324073
        {"n_estimators":2047,"max_features":0.4208871049610311,"max_leaves":45,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 2: metric=0.339302
        {"n_estimators":1770,"max_features":0.43265195304813925,"max_leaves":38,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 3: metric=0.340810
        {"n_estimators":752,"max_features":0.6891336674999727,"max_leaves":5981,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 4: metric=0.342603
        {"n_estimators":2047,"max_features":0.21564110265282438,"max_leaves":19,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 5: metric=0.343317
    ],

    # Top 5 configurations for lgbm
    'lgbm': [
        {"n_estimators":878,"num_leaves":316,"min_child_samples":4,"learning_rate":0.0033700699847521412,"log_max_bin":6,"colsample_bytree":0.8478876113343088,"reg_alpha":0.37765650632694386,"reg_lambda":0.04524306326267369,"FLAML_sample_size":49658},  # Rank 1: metric=0.332072
        {"n_estimators":145,"num_leaves":256,"min_child_samples":9,"learning_rate":0.0031321406232398324,"log_max_bin":6,"colsample_bytree":0.8994125388185525,"reg_alpha":0.009357561401367589,"reg_lambda":1.0626932216724168,"FLAML_sample_size":49658},  # Rank 2: metric=0.350842
        {"n_estimators":5,"num_leaves":64,"min_child_samples":125,"learning_rate":0.026822299108131054,"log_max_bin":5,"colsample_bytree":0.23304472552149,"reg_alpha":0.5127929178813916,"reg_lambda":0.06904243553895845,"FLAML_sample_size":49658},  # Rank 3: metric=0.354735
        {"n_estimators":472,"num_leaves":31,"min_child_samples":21,"learning_rate":0.0058625553385260225,"log_max_bin":7,"colsample_bytree":0.6850943442618156,"reg_alpha":1.5831652614415697,"reg_lambda":0.6600631771728843,"FLAML_sample_size":49658},  # Rank 4: metric=0.361147
        {"n_estimators":11,"num_leaves":128,"min_child_samples":114,"learning_rate":0.02517230877639018,"log_max_bin":6,"colsample_bytree":0.38996713838637465,"reg_alpha":5.4447913689079765,"reg_lambda":0.06384454446768235,"FLAML_sample_size":49658},  # Rank 5: metric=0.368505
    ],

    # Top 5 configurations for rf
    'rf': [
        {"n_estimators":347,"max_features":0.2701092383545975,"max_leaves":33,"criterion":'entropy',"FLAML_sample_size":49658},  # Rank 1: metric=0.346167
        {"n_estimators":1572,"max_features":0.382180762557646,"max_leaves":8,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 2: metric=0.346567
        {"n_estimators":106,"max_features":0.28751664963771006,"max_leaves":28,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 3: metric=0.350170
        {"n_estimators":1289,"max_features":0.5181979767528333,"max_leaves":27,"criterion":'entropy',"FLAML_sample_size":69943},  # Rank 4: metric=0.353080
        {"n_estimators":1597,"max_features":0.14642775303950256,"max_leaves":18,"criterion":'gini',"FLAML_sample_size":69943},  # Rank 5: metric=0.358196
    ],

    # Top 5 configurations for xgb_limitdepth
    'xgb_limitdepth': [
        {"n_estimators":88,"max_depth":7,"min_child_weight":4.0226033803073555,"learning_rate":0.004994733259560688,"subsample":0.6069709493711013,"colsample_bylevel":0.44378655429034874,"colsample_bytree":0.9741097693561845,"reg_alpha":0.11956298168467534,"reg_lambda":1.5146734481822137,"FLAML_sample_size":49658},  # Rank 1: metric=0.309307
        {"n_estimators":383,"max_depth":7,"min_child_weight":1.420929211420044,"learning_rate":0.0025148506288444586,"subsample":0.6407578925147996,"colsample_bylevel":0.5011723104767596,"colsample_bytree":0.82885576759612,"reg_alpha":0.021998672690898093,"reg_lambda":3.8814294819523854,"FLAML_sample_size":49658},  # Rank 2: metric=0.325609
        {"n_estimators":85,"max_depth":6,"min_child_weight":8.371600204320043,"learning_rate":0.0076228867067019606,"subsample":0.6950943179546162,"colsample_bylevel":0.4358012952803589,"colsample_bytree":1.0,"reg_alpha":0.005341494926689846,"reg_lambda":2.7163226990931078,"FLAML_sample_size":69943},  # Rank 3: metric=0.336252
        {"n_estimators":300,"max_depth":8,"min_child_weight":3.8388566513125997,"learning_rate":0.0037242707782056956,"subsample":0.6144490732929025,"colsample_bylevel":0.5231129038214993,"colsample_bytree":1.0,"reg_alpha":0.5953820179973823,"reg_lambda":4.450197773744569,"FLAML_sample_size":69943},  # Rank 4: metric=0.340427
        {"n_estimators":2567,"max_depth":7,"min_child_weight":7.4273842216318435,"learning_rate":0.004459977413947981,"subsample":0.5577532707820332,"colsample_bylevel":0.5287166712533805,"colsample_bytree":0.8736047240872873,"reg_alpha":0.039888525497280815,"reg_lambda":10.970652797339108,"FLAML_sample_size":49658},  # Rank 5: metric=0.344763
    ],

    # Top 5 configurations for xgboost
    'xgboost': [
        {"n_estimators":211,"max_leaves":9,"min_child_weight":3.121442885266257,"learning_rate":0.0035916684114108903,"subsample":0.8497081221021197,"colsample_bylevel":0.7980703079827159,"colsample_bytree":0.5081144092288252,"reg_alpha":0.08525587226921377,"reg_lambda":2.2387834336354495,"FLAML_sample_size":69943},  # Rank 1: metric=0.328736
        {"n_estimators":188,"max_leaves":4,"min_child_weight":2.9253384674362537,"learning_rate":0.004615356330910179,"subsample":0.7592994018354648,"colsample_bylevel":0.7040595247669855,"colsample_bytree":0.7071350532553218,"reg_alpha":0.6345188362523536,"reg_lambda":1.8858561362955628,"FLAML_sample_size":69943},  # Rank 2: metric=0.331753
        {"n_estimators":4688,"max_leaves":4,"min_child_weight":2.709682299917988,"learning_rate":0.0017004287097013885,"subsample":0.757964589830746,"colsample_bylevel":0.7599549631898695,"colsample_bytree":0.9932489721591821,"reg_alpha":13.18414663605857,"reg_lambda":7.035036708544174,"FLAML_sample_size":69943},  # Rank 3: metric=0.336490
        {"n_estimators":783,"max_leaves":4,"min_child_weight":9.350494144020972,"learning_rate":0.004595744807477995,"subsample":0.686234333351421,"colsample_bylevel":1.0,"colsample_bytree":0.9255126726060567,"reg_alpha":8.666292528504886,"reg_lambda":6.25607810857616,"FLAML_sample_size":69943},  # Rank 4: metric=0.336981
        {"n_estimators":1020,"max_leaves":11,"min_child_weight":4.793167651335496,"learning_rate":0.0011049825810174152,"subsample":0.9400573535360841,"colsample_bylevel":0.909686533766833,"colsample_bytree":0.4350857899655056,"reg_alpha":0.05974015683394514,"reg_lambda":2.260588404893929,"FLAML_sample_size":69943},  # Rank 5: metric=0.339506
    ],
}



"""
Train FLAML AutoML for hyperparameter optimization (training/optimization only).

This script performs FLAML AutoML training and logs all configurations to a log file.
It does NOT perform evaluation on test sets - use a separate script to parse the 
FLAML log file and extract the best configurations for later evaluation.

Usage:
    python train-flaml-topn.py <directory_path> [--time-budget MINUTES]

Example:
    python train-flaml-topn.py ./4-datasets/euos25_challenge_train_fluorescence340_450/desc2D+fp
    python train-flaml-topn.py ./4-datasets/euos25_challenge_train_fluorescence340_450/desc2D+fp --time-budget 120

Output:
    - Main log file with training progress and summary
    - FLAML log file (JSON) with all evaluated configurations and their performance
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import glob
import logging
import sys
import traceback
from collections import defaultdict
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    import flaml
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("ERROR: FLAML not available. Install with: pip install flaml")
    sys.exit(1)


def compute_effective_number_weights(y, beta=None):
    """
    Compute sample weights using Effective Number of Samples (ENS) approach.

    This method works better than sklearn's balanced weights for extreme class imbalance.
    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Args:
        y: Target labels (array-like)
        beta: Hyperparameter controlling the weighting strength (0 to 1)
              - Close to 1 (e.g., 0.9999): Strong re-weighting for extreme imbalance
              - Close to 0 (e.g., 0.9): Softer re-weighting
              - None: Auto-select based on class imbalance

    Returns:
        numpy array of sample weights
    """
    y_array = np.asarray(y).astype(int)
    class_counts = np.bincount(y_array)

    # Ensure we have at least 2 classes
    if len(class_counts) < 2:
        raise ValueError(f"Expected at least 2 classes, but found {len(class_counts)}")

    if beta is None:  # Auto-select beta based on imbalance
        positive_ratio = class_counts[1] / len(y_array)
        if positive_ratio < 0.01:  # Extreme (<1%)
            beta = 0.9999
        elif positive_ratio < 0.05:  # Severe (1-5%)
            beta = 0.999
        elif positive_ratio < 0.20:  # Moderate (5-20%)
            beta = 0.99
        else:  # Balanced (>20%)
            # Fall back to sklearn's balanced weights for relatively balanced data
            return compute_sample_weight('balanced', y_array)

    # Compute effective number of samples for each class
    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)

    # Compute weights per class (inverse of effective number)
    weights_per_class = 1.0 / effective_num

    # Normalize weights so they sum to number of classes
    weights_per_class = weights_per_class / weights_per_class.sum() * len(weights_per_class)

    # Assign weight to each sample based on its class
    sample_weights = np.array([weights_per_class[int(label)] for label in y_array])

    return sample_weights


def setup_logging(output_dir, dataset_name, descriptor_name):
    """Setup logging to both file and console."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = output_dir / f"optimization_{timestamp}.log"
    
    logger = logging.getLogger('train_flaml_topn')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename


def log_separator(logger, char='=', length=80):
    """Log a separator line."""
    logger.info(char * length)


def load_data(filepath, logger):
    """Load CSV file and handle NaN values."""
    try:
        logger.info(f"  Loading: {filepath}")
        df = pd.read_csv(filepath, compression='gzip')
        logger.info(f"    Shape: {df.shape}")
        
        # nan_count = df.isna().sum().sum()
        # if nan_count > 0:
        #     # logger.info(f"    Replacing {nan_count} NaN values with 0")
        #     # df = df.fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"    ERROR loading data: {str(e)}")
        raise


def prepare_data(df, logger):
    """Separate features and target."""
    try:
        if 'activity' not in df.columns:
            raise ValueError("'activity' column not found in dataframe")

        X = df.drop('activity', axis=1)
        y = df['activity']

        # Sanitize column names to remove special JSON characters that LightGBM doesn't support
        # Replace problematic characters with underscores
        original_cols = X.columns.tolist()
        sanitized_cols = []
        for col in original_cols:
            # Replace special JSON characters: " ' [ ] { } : , \ /
            sanitized = str(col)
            for char in ['"', "'", '[', ']', '{', '}', ':', ',', '\\', '/']:
                sanitized = sanitized.replace(char, '_')
            sanitized_cols.append(sanitized)

        X.columns = sanitized_cols

        # Check if any columns were changed
        changed_count = sum(1 for orig, san in zip(original_cols, sanitized_cols) if orig != san)
        if changed_count > 0:
            logger.info(f"    Sanitized {changed_count} column names to remove special JSON characters")

        class_counts = y.value_counts()
        logger.info(f"    Class distribution: {dict(class_counts)}")

        return X, y
    except Exception as e:
        logger.error(f"    ERROR preparing data: {str(e)}")
        raise


def train_flaml(X_train, y_train, time_budget_seconds, dataset_name, descriptor_name, output_dir, logger, use_weights=False, weight_strategy='ens'):
    """Train FLAML AutoML and return the automl object."""
    logger.info("  Initializing FLAML AutoML...")

    flaml_dir = Path(output_dir) / "flaml_logs"
    flaml_dir.mkdir(parents=True, exist_ok=True)

    # Create FLAML log file
    flaml_log_file = flaml_dir / "optimization.log"

    # Compute sample weights if requested
    sample_weights = None
    if use_weights:
        logger.info("\n  Analyzing class imbalance for sample weighting...")
        class_counts = y_train.value_counts().sort_index()
        positive_ratio = class_counts.get(1, 0) / len(y_train)
        logger.info(f"  Class distribution: {dict(class_counts)}")
        logger.info(f"  Positive class ratio: {positive_ratio:.4f} ({positive_ratio*100:.2f}%)")
        logger.info(f"  Unique classes in y_train: {sorted(y_train.unique())}")

        # Validate we have both classes
        if len(y_train.unique()) < 2:
            logger.error(f"  ERROR: Training data only contains {len(y_train.unique())} class(es): {list(y_train.unique())}")
            logger.error(f"  Cannot compute class weights with only one class. Disabling sample weighting.")
            use_weights = False

    if use_weights:
        if weight_strategy == 'ens':
            # Effective Number of Samples (ENS) - DEFAULT
            logger.info(f"  Weight strategy: EFFECTIVE NUMBER OF SAMPLES (ENS)")
            sample_weights = compute_effective_number_weights(y_train)

            # Log weight statistics
            unique_weights = np.unique(sample_weights)
            logger.info(f"  ENS weights computed:")
            if len(unique_weights) >= 2:
                logger.info(f"    Weight for class 0: {unique_weights[0]:.4f}")
                logger.info(f"    Weight for class 1: {unique_weights[1]:.4f}")
                logger.info(f"    Weight ratio (1/0): {unique_weights[1]/unique_weights[0]:.4f}")
            elif len(unique_weights) == 1:
                logger.info(f"    Single weight value: {unique_weights[0]:.4f}")
                logger.warning(f"    WARNING: Only one unique weight found. This may indicate an issue with weight calculation.")
            else:
                logger.error(f"    ERROR: No unique weights found!")

        elif weight_strategy == 'balanced':
            # Sklearn's balanced weights
            logger.info(f"  Weight strategy: SKLEARN BALANCED")

            if positive_ratio < 0.01:  # Extreme (<1%)
                sample_weights = compute_sample_weight('balanced', y_train)
                logger.info(f"  Using BALANCED weights (extreme imbalance)")
            elif positive_ratio < 0.05:  # Severe (1-5%)
                sample_weights = compute_sample_weight('balanced', y_train)
                logger.info(f"  Using BALANCED weights (severe imbalance)")
            elif positive_ratio < 0.20:  # Moderate (5-20%)
                sample_weights = compute_sample_weight('balanced', y_train)
                logger.info(f"  Using BALANCED weights (moderate imbalance)")
            else:  # Balanced (>20%)
                sample_weights = None
                logger.info(f"  No weights (data balanced)")

        elif weight_strategy == 'sqrt':
            # Square root scaled weights
            logger.info(f"  Weight strategy: SQRT-SCALED")

            if positive_ratio < 0.20:  # Apply only for imbalanced data
                imbalance_ratio = class_counts[0] / class_counts[1]
                sqrt_ratio = np.sqrt(imbalance_ratio)
                sample_weights = np.where(y_train == 1, sqrt_ratio, 1.0)
                logger.info(f"  Using SQRT-SCALED weights")
                logger.info(f"    Imbalance ratio: {imbalance_ratio:.2f}, Sqrt ratio: {sqrt_ratio:.2f}")
            else:  # Balanced (>20%)
                sample_weights = None
                logger.info(f"  No weights (data balanced)")
        else:
            logger.warning(f"  Unknown weight strategy: {weight_strategy}, using ENS as default")
            sample_weights = compute_effective_number_weights(y_train)
    else:
        logger.info("  Sample weighting: DISABLED (use_weights=False)")

    config = {
        'task': 'classification',
        'metric': 'roc_auc',
        'time_budget': time_budget_seconds,
        'log_file_name': str(flaml_log_file),
        'verbose': 1,
        'log_training_metric': True,
        'log_type': 'all',
        'early_stop': False,
        'seed': 42,
        'ensemble': False,
        'train_time_limit': 3000,
        'starting_points': warm_start_configs,
        'estimator_list': estimator_list,
    }

    # Add sample weights to config if computed
    if sample_weights is not None:
        config['sample_weight'] = sample_weights
        logger.info(f"  Sample weights added to FLAML config")

    automl = flaml.AutoML()

    logger.info("\n  Fitting FLAML model (this may take a while)...")
    logger.info(f"    Time budget: {time_budget_seconds}s ({time_budget_seconds/60:.1f} min)")
    logger.info(f"    FLAML log file: {flaml_log_file}")

    automl.fit(X_train, y_train, **config)

    logger.info("  Training complete!")

    return automl


def main():
    parser = argparse.ArgumentParser(
        description='Train FLAML AutoML (optimization only - no evaluation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train-flaml-topn.py ./4-datasets/euos25_challenge_train_fluorescence340_450/desc2D+fp
    python train-flaml-topn.py ./4-datasets/euos25_challenge_train_fluorescence340_450/desc2D+fp --time-budget 120

Note:
    This script only performs FLAML training/optimization and logs results.
    Use a separate script to parse the FLAML log file and extract best configurations for evaluation.
        """
    )
    parser.add_argument('directory', type=str, help='Directory containing train.csv.gz')
    parser.add_argument('--output-dir', type=str, default='./5-predictions/4-flaml-topn',
                        help='Output directory for FLAML logs (default: ./5-predictions/4-flaml-topn)')
    parser.add_argument('--time-budget', type=int, default=60,
                        help='Time budget in MINUTES for FLAML AutoML (default: 60)')
    parser.add_argument('--use-weights', action='store_true',
                        help='Enable sample weighting for imbalanced data (default: disabled)')
    parser.add_argument('--weight-strategy', type=str, default='ens',
                        choices=['ens', 'balanced', 'sqrt'],
                        help='Sample weighting strategy: ens (Effective Number of Samples - default), balanced (sklearn), sqrt (square root scaling)')

    args = parser.parse_args()
    
    if not FLAML_AVAILABLE:
        print("ERROR: FLAML is required but not available!")
        return 1
    
    directory_path = Path(args.directory)
    time_budget_seconds = args.time_budget * 60
    
    if directory_path.exists():
        dataset_name = directory_path.parent.name
        descriptor_name = directory_path.name
    else:
        dataset_name = "unknown"
        descriptor_name = "unknown"
    
    logger, log_filename = setup_logging(args.output_dir, dataset_name, descriptor_name)
    
    logger.info("="*80)
    logger.info("EUOS25 CHALLENGE - FLAML AUTOML TRAINING (OPTIMIZATION ONLY)")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")
    logger.info("")
    
    try:
        if not directory_path.exists():
            logger.error(f"ERROR: Directory does not exist: {directory_path}")
            return 1
        
        train_file = directory_path / 'train.csv.gz'
        
        if not train_file.exists():
            logger.error(f"ERROR: train.csv.gz not found in {directory_path}")
            return 1
        
        log_separator(logger)
        logger.info("CONFIGURATION")
        log_separator(logger)
        logger.info(f"Directory: {directory_path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Time budget: {args.time_budget} minutes ({time_budget_seconds} seconds)")
        logger.info(f"Train file: {train_file.name}")
        logger.info(f"Use weights: {args.use_weights}")
        if args.use_weights:
            logger.info(f"Weight strategy: {args.weight_strategy}")
        
        # Load data
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        log_separator(logger)
        df_train = load_data(train_file, logger)
        
        # Prepare data
        logger.info("\n" + "="*80)
        logger.info("PREPARING DATA")
        log_separator(logger)
        X_train, y_train = prepare_data(df_train, logger)
        
        logger.info(f"\n  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Check for constant columns
        logger.info("\nChecking for constant features...")
        constant_cols = X_train.columns[X_train.nunique() <= 1].tolist()
        if constant_cols:
            logger.info(f"  Found {len(constant_cols)} constant columns")
            logger.info(f"  Removing from train set...")
            X_train = X_train.drop(constant_cols, axis=1)
            logger.info(f"  After removal: {X_train.shape[1]} features")
        else:
            logger.info(f"  No constant columns found")
        
        # Train FLAML
        logger.info("\n" + "="*80)
        logger.info("TRAINING FLAML AUTOML")
        log_separator(logger)

        automl = train_flaml(X_train, y_train, time_budget_seconds, dataset_name, descriptor_name, args.output_dir, logger, use_weights=args.use_weights, weight_strategy=args.weight_strategy)
        
        # Log FLAML summary
        logger.info("\n" + "="*80)
        logger.info("FLAML TRAINING SUMMARY")
        log_separator(logger)
        logger.info(f"  Best estimator: {automl.best_estimator}")
        logger.info(f"  Best validation loss: {automl.best_loss:.6f}")
        logger.info(f"  Best validation AUROC: {1.0 - automl.best_loss:.6f}")
        logger.info(f"  Total models evaluated: {len(automl.config_history)}")
        logger.info(f"  Best config: {automl.best_config if hasattr(automl, 'best_config') else 'N/A'}")
        
        log_separator(logger)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        log_separator(logger)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"All configurations logged to FLAML log file")
        logger.info(f"Use a separate script to parse the log and extract best configurations")
        logger.info("")
        
        return 0
    
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("FATAL ERROR")
        logger.error("="*80)
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    exit(main())


