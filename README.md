# ASTBasedMetrics4CSDetector
-----
A method of counting three types of AST-based code metrics to help identify code smells.
-----
1. We provide our experimental environment file "py36.yaml", open the Terminal, and execute the command line to copy our experimental environment quickly: "conda env create -f py36.yaml". Execute "conda activate py36" to activate the environment.
2. Before running all source codes, please modify the paths in the code correctly.
3. Execute "python machineLearningDetectionX5.py" and "python deepLearningDetectionX5.py" to obtain the experimental results of RQ1~3. Similarly, execute "python machineLearningDetectionX5-RQ4.py" and "python deepLearningDetectionX5-RQ4.py" to get the experimental results of RQ4. Before running, you need to modify the path and parameter settings, such as "our_RQ = 1, binaryClassifaction = True".
4. We also provide three types of AST-based code metric calculation methods and five feature selection algorithms, which you can freely explore and test.
