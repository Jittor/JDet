import jittor as jt 

HEADER = r'''
#include <cfloat>

#define CeilDIV(a,b) ((a+b-1)/b)
// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void SigmoidFocalLossForward(const int nthreads,
                                        const scalar_t *logits,
                                        const int *targets,
                                        const int num_classes,
                                        const float gamma, const float alpha,
                                        const int num, scalar_t *losses) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int n = i / num_classes;
    int d = i % num_classes;  // current class[0~79];
    int t = targets[n];       // target class [1~80];

    // Decide it is positive or negative case.
    scalar_t c1 = (t == (d + 1));
    scalar_t c2 = (t >= 0 & t != (d + 1));

    scalar_t zn = (1.0 - alpha);
    scalar_t zp = (alpha);

    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    scalar_t p = 1. / (1. + expf(-logits[i]));

    // (1-p)**gamma * log(p) where
    scalar_t term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));

    // p**gamma * log(1-p)
    scalar_t term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;

  }  // CUDA_1D_KERNEL_LOOP
}  // SigmoidFocalLossForward

template <typename scalar_t>
__global__ void SigmoidFocalLossBackward(
    const int nthreads, const scalar_t *logits, const int *targets,
    const scalar_t *d_losses, const int num_classes, const float gamma,
    const float alpha, const int num, scalar_t *d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int n = i / num_classes;
    int d = i % num_classes;  // current class[0~79];
    int t = targets[n];       // target class [1~80], 0 is background;

    // Decide it is positive or negative case.
    scalar_t c1 = (t == (d + 1));
    scalar_t c2 = (t >= 0 & t != (d + 1));

    scalar_t zn = (1.0 - alpha);
    scalar_t zp = (alpha);
    // p = 1. / 1. + expf(-x); p = sigmoid(x)
    scalar_t p = 1. / (1. + expf(-logits[i]));

    // (1-p)**g * (1 - p - g*p*log(p)
    scalar_t term1 =
        powf((1. - p), gamma) * (1. - p - (p * gamma * logf(max(p, FLT_MIN))));

    // (p**g) * (g*(1-p)*log(1-p) - p)
    scalar_t term2 =
        powf(p, gamma) *
        ((-1. * logits[i] * (logits[i] >= 0) -
          logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
             (1. - p) * gamma -
         p);
    d_logits[i] = 0.0;
    d_logits[i] += -c1 * term1 * zp;
    d_logits[i] += -c2 * term2 * zn;
    d_logits[i] = d_logits[i] * d_losses[i];

  }  // CUDA_1D_KERNEL_LOOP
}  // SigmoidFocalLossBackward
'''

FORWARD_SRC=r'''
@alias(logits,in0)
@alias(targets,in1)
@alias(losses,out0)

const int num_samples = logits_shape0;
int losses_size = num_samples * logits_shape1;

dim3 grid(std::min(CeilDIV(losses_size, 512), 4096));
dim3 block(512);


SigmoidFocalLossForward<<<grid, block>>>(
            losses_size, logits_p,
            targets_p, num_classes, gamma, alpha,
            num_samples, losses_p);

'''

BACKWARD_SRC=r'''
@alias(logits,in0)
@alias(targets,in1)
@alias(d_losses,in2)
@alias(d_logits,out0)

 const int num_samples = logits_shape0;

  int d_logits_size = num_samples * logits_shape1;

  dim3 grid(std::min(CeilDIV(d_logits_size, 512),4096));
  dim3 block(512);

  SigmoidFocalLossBackward<<<grid, block>>>(
      d_logits_size, logits_p,
      targets_p,
      d_losses_p, num_classes, gamma, alpha,
      num_samples, d_logits_p);
'''
def sigmoid_focal_loss_forward(input, target,num_classes, gamma, alpha):
    # print(jt.misc.unique(target))
    assert target.max().item()<=input.shape[1]
    i_src = f"const int num_classes = {num_classes}; const float gamma={gamma};const float alpha = {alpha};"
    return jt.code(input.shape,input.dtype,[input,target],cuda_header=HEADER,cuda_src=i_src+FORWARD_SRC)

def sigmoid_focal_loss_backward(input, target,d_losses,num_classes,gamma, alpha):
    i_src = f"const int num_classes = {num_classes}; const float gamma={gamma};const float alpha = {alpha};"
    return jt.code(input.shape,input.dtype,[input,target,d_losses],cuda_header=HEADER,cuda_src=i_src+BACKWARD_SRC)

class SigmoidFocalLossFunction(jt.Function):

    def execute(self, input, target, gamma=2.0, alpha=0.25):
        self.saved_tensors = (input, target)
        num_classes = input.shape[1]
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha

        loss = sigmoid_focal_loss_forward(input, target, num_classes,
                                               gamma, alpha)
        return loss

    def grad(self, d_loss):
        input, target = self.saved_tensors
        num_classes = self.num_classes
        gamma = self.gamma
        alpha = self.alpha
        d_input = sigmoid_focal_loss_backward(input, target, d_loss,
                                                   num_classes, gamma, alpha)
        return d_input


sigmoid_focal_loss = SigmoidFocalLossFunction.apply
