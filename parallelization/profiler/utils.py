
def log_parameter_count(model, model_args):
    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32, 4 bytes per param
    
    print(f"📊 Model Statistics:")
    print(f"   🔢 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    print(f"   💾 Model size: {param_size_mb:.2f} MB")
    print(f"   🧮 Model config: dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")