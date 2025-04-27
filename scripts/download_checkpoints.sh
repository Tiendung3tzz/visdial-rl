
mkdir -p /kaggle/working/visdial-rl/checkpoints
cd /kaggle/working/visdial-rl/checkpoints

gdown --id 1ONO67T_fcSKF8PKSUDiuyqSzN0ux4WRX -O abot_sl_ep60.vd
gdown --id 1JDEqHKVSL4-CZ4r9RgXjBRcfruQ1Jk3C -O qbot_sl_ep60.vd

gdown --id 15jcIynNT4S8OqdS6ZibuC7kXK3CMA4YJ -O abot_rl_ep20.vd
gdown --id 16B6Cf93W7N8QDbnDbFkAvuvcAFlgxZQu -O qbot_rl_ep20.vd