var layers = {
    'Linear': (N, in_features, out_features) => {return [N, out_features]},
    
    'ReLU': (N, shape) => {return [N, shape]},
    
    'Conv2d': (N, C_in, H, W, C_out, kernel_size, stride=1, padding=0, dilation=1) => {
      return [
        N, C_out, 
        Math.floor((H + 2*padding - dilation * (kernel_size-1) -1)/stride + 1), 
        Math.floor((W + 2*padding - dilation * (kernel_size-1) -1)/stride + 1)
      ]
    },
  
    'MaxPool2d': (N, C, H, W, kernel_size, stride, padding=0, dilation=1) => {
      return layers.Conv2d(N, C, H, W, C, kernel_size, stride, padding, dilation)
    },
  
    'ConvTranspose2d': (N, C_in, H, W, C_out, kernel_size, stride=1, padding=0, output_padding=0) => {
      return [
        N, C_out,
        Math.floor((H-1)*stride - 2*padding + kernel_size + output_padding),
        Math.floor((W-1)*stride - 2*padding + kernel_size + output_padding),
      ]
    },
  
    'Flatten': (N, C_in, H, W) => {
      return [N, C_in * H * W]
    },
  
    'Reshape': (N, in_features, C_out, H, W) => {
      if(in_features == C_out * H * W)
        return [N, C_out, H, W]
      else {
        console.log(
          "Could not reshape input of size " + in_features + " to (" + C_out + ", " + H + ", " + W + ") which would be " + C_out*H*W
        )
        return null
      }
    }
  }
  
  x = [1, 3, 64, 64]
  console.log("Input: " + x)
  x = layers.Conv2d.apply(null, x.concat([128, 3, 3, 1, 1]))
  console.log("Conv2d: " + x)
  
  x = layers.Conv2d.apply(null, x.concat([64, 3, 2, 1, 1]))
  console.log("Conv2d: " + x)
  
  x = layers.Conv2d.apply(null, x.concat([32, 3, 2, 1, 1]))
  console.log("Conv2d: " + x)
  
  x = layers.Flatten.apply(null, x)
  console.log("Flatten:" + x)
  
  x = layers.Linear.apply(null, x.concat(2048))
  console.log("Linear: " + x)
  x = layers.Linear.apply(null, x.concat(10))
  console.log("Linear: " + x)
  x = layers.Linear.apply(null, x.concat(2048))
  console.log("Linear: " + x)
  x = layers.Linear.apply(null, x.concat(1152))
  console.log("Linear: " + x)
  
  x = layers.Reshape.apply(null, x.concat(32, 6, 6))
  console.log("Reshape: " + x)
  
  x = layers.ConvTranspose2d.apply(null, x.concat(64, 3, 2, 1, 0))
  console.log("ConvTranspose2d: " + x)
  
  x = layers.ConvTranspose2d.apply(null, x.concat(128, 3, 2, 1, 1))
  console.log("ConvTranspose2d: " + x)
  
  //kernel_size, stride=1, padding=0, output_padding=0
  x = layers.ConvTranspose2d.apply(null, x.concat(3, 3, 3, 1, 0))
  console.log("ConvTranspose2d: " + x)