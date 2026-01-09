from models.model_tools import IntModelFromSpec
from data.data_loader import load_data_sample
from analysis.visualizer import ConfusionMatrixRecorder, ConfusionMatrixWriter
import torch
import torch.nn as nn

def train_model(
        number_of_epochs, 
        batch_size, 
        sample_size, 
        dataset_path, 
        spec, 
        optimizer = None,
        loss_function = None,
        record_confusion_matrices = False, 
        output_dir = None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data_sample(dataset_path, sample_size=sample_size, batch_size=batch_size)
    model = IntModelFromSpec(spec).to(device)
    if optimizer is not None:
        optimizer._register(model)
    
    if (record_confusion_matrices):
        confusion_recorder = ConfusionMatrixRecorder(num_classes=spec[3])
        confusion_writer = ConfusionMatrixWriter(num_classes=spec[3], output_dir=output_dir)


    for epoch in range(number_of_epochs):
        total_correct = 0
        total_samples = 0
        logit_accumulation = 0
        if record_confusion_matrices:
            confusion_recorder.reset()
    
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits = model(x)
                logit_accumulation += logits.to(torch.int32).abs().sum().item()
                preds = torch.argmax(logits, dim=1)

                if loss_function is not None:
                    error = loss_function.compute(logits, y)
                else :
                    error = torch.zeros_like(logits, dtype=torch.int32, device=logits.device)

                model.backward(error)

                if optimizer is not None:
                    optimizer.step(model)

                if record_confusion_matrices:
                    confusion_recorder.update(preds, y)

                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        if record_confusion_matrices:
            cm_norm = confusion_recorder.compute()
            acc = total_correct / total_samples if total_samples > 0 else 0
            confusion_writer.write(cm_norm, epoch, acc)
            print(f"Epoch {epoch:03d} | Acc: {acc:.4f}")
        average_logit = logit_accumulation / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch:03d} | Average Logit Magnitude {average_logit:.4f}")
    if record_confusion_matrices:
        confusion_writer.close()