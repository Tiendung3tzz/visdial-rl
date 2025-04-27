# #!/usr/bin/env bash

# # SL checkpoints
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep60.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep60.vd

# # SL-Delta checkpoints
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep15_delta.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep15_delta.vd


# # RL checkpoints
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20.vd

# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20.vd

# # RL-Delta checkpoints
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10_delta.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20_delta.vd

# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10_delta.vd
# curl -o checkpoints/ https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20_delta.vd

#!/usr/bin/env bash

# Ensure the checkpoints directory exists
mkdir -p checkpoints

# SL checkpoints
curl -o checkpoints/abot_sl_ep60.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep60.vd
curl -o checkpoints/qbot_sl_ep60.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep60.vd

# SL-Delta checkpoints
curl -o checkpoints/abot_sl_ep15_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_sl_ep15_delta.vd
curl -o checkpoints/qbot_sl_ep15_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_sl_ep15_delta.vd

# RL checkpoints
curl -o checkpoints/abot_rl_ep10.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10.vd
curl -o checkpoints/abot_rl_ep20.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20.vd
curl -o checkpoints/qbot_rl_ep10.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10.vd
curl -o checkpoints/qbot_rl_ep20.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20.vd

# RL-Delta checkpoints
curl -o checkpoints/abot_rl_ep10_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep10_delta.vd
curl -o checkpoints/abot_rl_ep20_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/abot_rl_ep20_delta.vd
curl -o checkpoints/qbot_rl_ep10_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep10_delta.vd
curl -o checkpoints/qbot_rl_ep20_delta.vd https://s3.amazonaws.com/cvmlp/visdial-pytorch/models/qbot_rl_ep20_delta.vd
