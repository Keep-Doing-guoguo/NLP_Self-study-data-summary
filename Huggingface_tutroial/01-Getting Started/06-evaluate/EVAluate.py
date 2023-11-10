import evaluate


a = evaluate.list_evaluation_modules(include_community=False, with_details=True)

a1 = accuracy = evaluate.load("accuracy")


print(accuracy.description)


print(accuracy.inputs_description)


accuracy = evaluate.load("accuracy")
results = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])

accuracy = evaluate.load("accuracy")#下面这段是和上面的性质是相同的
for ref, pred in zip([0,1,0,1], [1,0,0,1]):
    accuracy.add(references=ref, predictions=pred)
a2 = accuracy.compute()


accuracy = evaluate.load("accuracy")
for refs, preds in zip([[0,1],[0,1]], [[1,0],[0,1]]):
    accuracy.add_batch(references=refs, predictions=preds)
a3 = accuracy.compute()


clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])#将会计算三种

a4 = clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])


from evaluate.visualization import radar_plot   # 目前只支持雷达图

data = [
   {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
   {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
   {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6},
   {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_names = ["Model 1", "Model 2", "Model 3", "Model 4"]

plot = radar_plot(data=data, model_names=model_names)
print()


































