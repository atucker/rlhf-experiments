accelerate launch --num_processes 2 src/dips/tldr/kl.py
accelerate launch --num_processes 2 src/dips/tldr/kl.py --train_dips --factor_loss 
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --train_dips --factor_loss 
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --loss_full_precision
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --train_dips --factor_loss --loss_full_precision
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --loss_full_precision --unembed_full_precision
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --train_dips --factor_loss --loss_full_precision --unembed_full_precision
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --unembed_full_precision
accelerate launch --num_processes 2 --mixed_precision bf16 src/dips/tldr/kl.py --train_dips --factor_loss--unembed_full_precision