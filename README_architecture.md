PyTorch expects the following format for an input:
(batch_size, n_channels, height, width)
for an image. The original code resized the embeddings inputs to dimensions of:
(batch_size, 3, 256, 128)

It makes sense to make the height-to-width aspect ration (2/1) because we expect the height of a body to be much longer (we assume twice as long) as the width of the body. 

However, once we pre-process the embedding input to above dimensions on the `siamese_net.py`, we attain below dimensions:
```
torch.Size([1024, 3])
```

This is incorrect as we need our embeddings to be a vector, not a 2d matrix. To remedy this, we could resize the input to be of dimension:
```
(batch_size, 3, 128, 128)
```
This will result in a 1024-dimensional vector. The problem with this approach is that we have a low-resolution input. Our embedding is able to create a more robust separation between different identities if it provided a richer expression of the identity (which can be done by increasing the resolution of the input). Further, the aspect ration now becomes 1:1, we assume that the height and width of a human are roughly equal. 

As such, to retain the original resizing of the image, we modify the architecture by updating the last convolution to be:
```
nn.Conv2d(512,1024,kernel_size=(3,1),stride=1)
```
Because the input to the above layer is `torch.Size([1, 512, 3, 1])`, the kernel is equal to the (height,width) dimensions of our input. Thus, the convolution becomes equivalent to the following dense-operation:
```
nn.Linear(512*3*1, 1024)(input.view(1,512*3*1))
```