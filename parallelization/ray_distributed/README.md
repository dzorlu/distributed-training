



```
source ~/.bashrc
echo $RAY_HEAD_IP    # Should show the head node IP
echo $RAY_ADDRESS    # Should show head_ip:6379

# Connect to Ray
python3 -c "import ray; ray.init(address='$RAY_ADDRESS'); print(ray.nodes())"
```