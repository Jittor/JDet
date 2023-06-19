_base_ = ['./redet_re50_refpn_1x_dota.py']

dataset=dict(
    train=dict(balance_category=True)
)
