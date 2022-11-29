import time
import copy
import paddle


# Create a key-only dictionary
def generate_metrics_list(metrics_def):
    rlist = {}
    for name in metrics_def.keys():
        rlist[name] = []
    return rlist


# Epoch kernel for both [Validation Set, training=False] and [Training Set, training=True]
def epoch(scope, dataSetLoader, on_batch=None, training=False):
    model = scope["model"]
    optimizer = scope["optimizer"]
    loss_func = scope["loss_func"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)
    scope["loader"] = dataSetLoader
    metrics_list = generate_metrics_list(metrics_def)
    total_loss = 0

    if training:
        model.train()  # Training Set
    else:
        model.eval()  # Validation Set
        # 使用GPU进行训练

    with paddle.static.device_guard('gpu'):  # 该 API 仅支持静态图模式
        for tensors in dataSetLoader:
            if "process_batch" in scope and scope["process_batch"] is not None:
                tensors = scope["process_batch"](tensors)

            if "device" in scope and scope["device"] is not None:  # device == 0d
                tensors = [tensor.to(scope["device"]) for tensor in tensors]  # tensor.to(0) can't explain why this works

            loss, output = loss_func(model, tensors)

            if training:
                optimizer.clear_grad()  # This [optimizer] can be treated as a function based on gradient value(GD).
                loss.backward()  # backward() calculates the gradient value
                optimizer.step()  # Update parameters by running the [optimizer] once, valid only for [Dygraph].
            total_loss += loss.item()  # Update total loss
            scope["batch"] = tensors
            scope["loss"] = loss
            scope["output"] = output
            scope["batch_metrics"] = {}
            for name, metric in metrics_def.items():
                value = metric["on_batch"](scope)
                scope["batch_metrics"][name] = value
                metrics_list[name].append(value)
            if on_batch is not None:
                on_batch(scope)
    scope["metrics_list"] = metrics_list
    metrics = {}
    for name in metrics_def.keys():
        scope["list"] = scope["metrics_list"][name]
        metrics[name] = metrics_def[name]["on_epoch"](scope)
    return total_loss, metrics


# Training kernel
def train(scope, train_dataset, val_dataset, batch_size=256, eval_model=None,
          on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    skips = 0  # If the model is not the best in loss
    model = scope["model"]
    epochs = scope["epochs"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_model"] = None
    scope["best_val_metrics"] = None
    scope["best_train_metric"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_train_loss"] = float("inf")

    localTime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    fileName = localTime + ' TrainLog.txt'

    # Training Kernel Loop
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        with open(fileName, "a") as f:
            print("Epoch #" + str(epoch_id))  # 1
            f.write("Epoch #" + str(epoch_id) + "\n")

            # Training
            scope["dataset"] = train_dataset
            train_loss, train_metrics = epoch(scope, train_loader, on_train_batch, training=True)
            scope["train_loss"] = train_loss
            scope["train_metrics"] = train_metrics
            print("\tTrain Loss = " + str(train_loss))  # 1
            f.write("\tTrain Loss = " + str(train_loss) + "\n")
            for name in metrics_def.keys():
                print("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]))
                f.write("\tTrain " + metrics_def[name]["name"] + " = " + str(train_metrics[name]) + "\n")
            if on_train_epoch is not None:
                on_train_epoch(scope)
            del scope["dataset"]

            # Validation
            scope["dataset"] = val_dataset
            with paddle.no_grad():  # Substitute torch by paddle
                val_loss, val_metrics = epoch(scope, val_loader, on_val_batch, training=False)
            scope["val_loss"] = val_loss
            scope["val_metrics"] = val_metrics
            print("\tValidation Loss = " + str(val_loss))
            f.write("\tValidation Loss = " + str(val_loss) + "\n")
            for name in metrics_def.keys():
                print("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]))
                f.write("\tValidation " + metrics_def[name]["name"] + " = " + str(val_metrics[name]) + "\n")
            if on_val_epoch is not None:
                on_val_epoch(scope)
            del scope["dataset"]

            # Selection
            is_best = None
            if eval_model is not None:
                is_best = eval_model(scope)
            if is_best is None:
                is_best = val_loss < scope["best_val_loss"]
            if is_best:
                scope["best_train_metric"] = train_metrics
                scope["best_train_loss"] = train_loss
                scope["best_val_metrics"] = val_metrics
                scope["best_val_loss"] = val_loss
                scope["best_model"] = copy.deepcopy(model)
                print("Model saved!")
                f.write("Model saved!" + "\n")
                if epoch_id > 500:
                    paddle.save(model.state_dict(), "DeepCFD_" + str(epoch_id) + ".pdparams")
                    print("Model saved!")
                    f.write("Model saved!" + "\n")
                skips = 0
                skips = 0
            else:
                skips += 1
            if after_epoch is not None:
                after_epoch(scope)
    return scope["best_model"], scope["best_train_metric"], scope["best_train_loss"], scope["best_val_metrics"], scope["best_val_loss"]


# val_dataset == test_dataset
# Preparation before the Training and Printing
def train_model(model, loss_func, train_dataset, val_dataset, optimizer, process_batch=None, eval_model=None,
                on_train_batch=None, on_val_batch=None, on_train_epoch=None, on_val_epoch=None, after_epoch=None,
                epochs=100, batch_size=256, device=0, **kwargs):
    # model = model.to(device) # Set device for pytorch
    # [scope] is a dictionary storing many training parameters
    scope = {"model": model, "loss_func": loss_func, "train_dataset": train_dataset, "val_dataset": val_dataset,
             "optimizer": optimizer, "process_batch": process_batch, "epochs": epochs, "batch_size": batch_size,}

    # For Printing
    names = []  # A list for building dictionary named [metrics_def]
    metrics_def = {}  # A dictionary for printing
    for key in kwargs.keys():  # Put strings ["m_ux_name"] ["m_uy_name"] ["m_up_name"] in dictionary [names], weird
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])

    for name in names:  # A dictionary for printing
        if "m_" + name + "_name" in kwargs and "m_" + name + "_on_batch" in kwargs and "m_" + name + "_on_epoch" in kwargs:
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                # For example, to handle Ux, put ["Ux MSE"] in new dic metrics_def by name ['ux']
                "on_batch": kwargs["m_" + name + "_on_batch"],
                # For example, to handle Ux, put [lamda function] in new dic metrics_def by name ['ux']
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
                # For example, to handle Ux, put [lamda function] in new dic metrics_def by name ['ux']
            }
        else:
            print("Warning: " + name + " metric is incomplete!")  # Check if u miss any mse in epochs or batch
    scope["metrics_def"] = metrics_def
    # For Printing Ended

    return train(scope, train_dataset, val_dataset, eval_model=eval_model, on_train_batch=on_train_batch,
                 on_val_batch=on_val_batch, on_train_epoch=on_train_epoch, on_val_epoch=on_val_epoch,
                 after_epoch=after_epoch, batch_size=batch_size)
