import torch
import horovod.torch as hvd

# Initialize Horovod for multi-GPU training
hvd.init()
torch.cuda.set_device(hvd.local_rank())

# Define a simple model
model = torch.nn.Linear(10, 1).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = hvd.DistributedOptimizer(optimizer)

for epoch in range(10):
    output = model(torch.randn(100, 10).cuda())
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if hvd.rank() == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
