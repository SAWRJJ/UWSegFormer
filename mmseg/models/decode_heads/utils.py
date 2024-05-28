#!/usr/bin/env python3

import torch.nn as nn


def init_loss(loss, builder, num_classes=None):
    """Helper function to initialize loss (multilabel or single)."""
    if isinstance(loss, dict):
        if ("num_classes" in loss.keys()) and num_classes:
            loss.update(dict(num_classes=num_classes))
        out_loss = builder(loss)
    elif isinstance(loss, (list, tuple)):
        out_loss = nn.ModuleList()
        for l in loss:
            if ("num_classes" in l.keys()) and num_classes:
                l.update(dict(num_classes=num_classes))
            out_loss.append(builder(l))
    else:
        if loss is None:
            out_loss = None
        else:
            raise TypeError(
                f"loss must be a dict or sequence of dict or None,\
                but got {type(loss)}"
            )
    return out_loss


def get_loss_names(losses):
    loss_names = []

    if not isinstance(losses, (list, tuple)):
        losses = [losses]

    for loss in losses:
        if loss is not None:
            if not isinstance(loss, nn.ModuleList):
                loss_names.append(loss.loss_name)
            else:
                for l in loss:
                    loss_names.append(l.loss_name)

    return loss_names
