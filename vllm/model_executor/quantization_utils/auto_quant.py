from typing import NamedTuple, Optional, TypeVar, Union, overload
import torch
from torch import Tensor, device, dtype

try:
    from vllm import quantization_ops
except ImportError:
    pass


T = TypeVar("T", bound="torch.nn.Module")

class QParams(NamedTuple):
    """A class to hold the quantization parameters."""

    scales: torch.Tensor
    zero_points: Optional[torch.Tensor]


@torch.no_grad()
def cal_qparams_per_group_minmax(w: torch.Tensor,
                                 n_bits: int = 4,
                                 group_size: int = 128):
    """Calculate quantization parameters for each group using min and max
    values."""

    outc, inc = w.shape
    assert inc >= group_size, \
        'Input channels should be greater than or equal to group_size.'
    assert inc % group_size == 0, \
        'Input channels should be divisible by group_size.'
    w_group_wise = w.reshape(outc, -1, group_size)
    w_min = w_group_wise.min(dim=-1, keepdim=True)[0]
    w_max = w_group_wise.max(dim=-1, keepdim=True)[0]

    q_max = 2**n_bits - 1
    q_min = 0
    scales = (w_max - w_min)
    scales = scales.clamp_(min=1e-5).div_(q_max)
    # zero_points = (-w_min / scales).round().clamp(q_min, q_max)
    zero_points = (-torch.round(w_min / scales)).clamp_(q_min, q_max)
    return QParams(scales=scales, zero_points=zero_points)


def convert_s4(qw: torch.Tensor, qz: torch.Tensor, s: torch.Tensor,
               group_size: int = 128):
    assert qw.is_contiguous()
    assert qz.is_contiguous()
    assert s.is_contiguous()
    _qw = torch.zeros_like(qw)
    _sz = torch.zeros_like(s, dtype=torch.int32)  # half2
    _ws = torch.zeros_like(s)
    quantization_ops.ops_convert_s4_k_m8(_qw, _sz, _ws, qw, s, qz,
                        qw.size(-1) * 8, qw.size(0), group_size)
    return _qw, _sz


def tp_m_s4(x: torch.Tensor, tp: int = 1):
    return x.view(x.size(0) // 32, tp, -1, 128).permute(0, 2, 3,
                                                            1).contiguous()


def quant(weight: torch.Tensor,
          qparams: Optional[QParams] = None) -> torch.Tensor:
    """Perform quantization on the given weight tensor.
    Args:
        weight (torch.Tensor): The weight tensor with shape
            (out_features, in_features).
        qparams (Optional[QParams]): A namedtuple containing 'scales'
            and 'zero_points'.
    Returns:
        torch.Tensor: The fake quantized weight tensor.
    """
    if qparams is None:
        qparams = cal_qparams_per_group_minmax(weight)
    scales = qparams.scales
    zero_points = qparams.zero_points
    out_c, in_c = weight.shape
    # Reshape the weights if using per_group quantization
    # per tensor scales shape: [1]
    # per channel scales shape: [out_c, 1]
    # per group scales shape: [out_c, in_c//group_size, 1]
    if len(scales.shape) > 2:
        # scales shape: [out_c, in_c//group_size, 1]
        weight = weight.reshape(out_c, scales.shape[1], -1)
    if zero_points is None:
        real_qweight = (weight / scales).round()
    else:
        real_qweight = ((weight + (scales * zero_points)) / scales).round()
    if len(scales.shape) > 2:
        real_qweight = real_qweight.reshape(out_c, in_c)
    return real_qweight.to(torch.int32)


def quantize_tensor(weight, n_bits=4, group_size=128):
    pack_num = 32 // n_bits
    pack_order = [0, 2, 4, 6, 1, 3, 5, 7]
    org_weight_shape = weight.shape
    out_features = org_weight_shape[0]
    in_features = org_weight_shape[1]
    print(f'weight.shape: {weight.shape}, out_features:{out_features}, in_features: {in_features}')
    qparams = cal_qparams_per_group_minmax(weight, n_bits)
    i32_w = quant(weight, qparams)
    i32_w = i32_w.t().contiguous()
    w_pack_oc = out_features // (32 // n_bits)
    w_inc = in_features
    pack_int_w = torch.zeros((w_inc, w_pack_oc), dtype=torch.int32, device=weight.device)
    for col in range(pack_int_w.shape[1]):
        for i in range(pack_num):
            pack_int_w_col = i32_w[:, col * pack_num + pack_order[i]]
            pack_int_w[:, col] |= pack_int_w_col << (i * n_bits)
    qweight = pack_int_w
    scales = qparams.scales.squeeze(-1).t().contiguous()
    if qparams.zero_points is not None:
        zeros = qparams.zero_points.to(torch.int32)
        zeros = zeros.squeeze(-1).t().contiguous()
        z_inc = in_features // group_size
        z_oc = out_features // (32 // n_bits)
        pack_int_zeros = torch.zeros((z_inc, z_oc), dtype=torch.int32, device=weight.device)
        for col in range(pack_int_zeros.shape[1]):
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + pack_order[i]]
                pack_int_zeros[:, col] |= qzero_col << (i * n_bits)
        qzeros = pack_int_zeros
        print(f'qweight.shape: {qweight.shape}, scales: {scales.shape}, qzeros: {qzeros.shape}')
        print(f'qweight.dtype: {qweight.dtype}, scales: {scales.dtype}, qzeros: {qzeros.dtype}')
    return qweight, scales, qzeros


class Int4Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        scales_zeros=None
    ):
        cls.scales_zeros = None
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data)

    def cuda(self, device):
        # we store the 4-bit rows-major weight
        # we convert this weight to the turning/ampere weight during the first inference pass
        data_fp = self.data.contiguous().half().cuda(device)
        _qweight, _scales, _qzeros = quantize_tensor(
            data_fp, n_bits=4, group_size=128)
        qweight, scales_zeros = convert_s4(_qweight, _qzeros, _scales)
        self.data = qweight
        setattr(self, "scales_zeros", scales_zeros)

        return self

    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if (
            device is not None
            and device.type == "cuda"
            and self.data.device.type == "cpu"
        ):
            return self.cuda(device)
        else:
            new_param = Int4Params(
                super().to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                ),
                scales_zeros=self.scales_zeros
            )
            return new_param