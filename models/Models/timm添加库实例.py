import timm as timm
print(timm.list_models())
print(timm.list_modules())
""" 引用普通的模块"""
conv = timm.models.layers.nn.Conv2d
print(conv)
""" 引用注意力模块
'conv att:'
ese eca ecam ecam ceca ge gc gca cbam lcbam sk splat
'self-attention:'
lambda bottleneck halo nl bat
"""
ATTention = timm.models.layers.create_attn('se', 512)
print(ATTention)
""" 引用act模块"""

""" 引用卷积模块"""

""" 引用norm模块"""

""" 引用optim模块"""


