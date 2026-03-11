# Apply adaptive learning rate adjustment based on KL divergence
        if args.target_kl is not None:
            global_approx_kl = approx_kl
            if global_approx_kl > 2.0 * args.target_kl:
                alpha = max(1e-5, alpha / 1.5)
            elif global_approx_kl < 0.5 * args.target_kl:
                # cap the learning rate to prevent divergence 
                # After 8M steps, the learning rate is around 5e-4
                # So we reduce the cap to 8e-4 to prevent divergence
                if global_step > 8_000_000:
                    alpha = min(8e-4, alpha * 1.5)
                else:
                    alpha = min(1e-2, alpha * 1.5)