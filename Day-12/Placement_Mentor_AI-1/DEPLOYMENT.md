# AWS EC2 Deployment Guide — PlacementMentorAI

## Prerequisites

- AWS Account with EC2 access
- SSH key pair configured
- Basic understanding of AWS Security Groups

---

## Step 1: Launch EC2 Instance

### Recommended Instance
| Setting | Value |
|---------|-------|
| **AMI** | Ubuntu 22.04 LTS |
| **Instance Type** | `t3.xlarge` (4 vCPU, 16 GB RAM) or `g4dn.xlarge` for GPU |
| **Storage** | 30 GB gp3 SSD minimum |
| **Key Pair** | Your SSH key pair |

> **Note**: YOLO26-Pose runs on CPU but is significantly faster on GPU. For production workloads, use a `g4dn.xlarge` instance with NVIDIA T4 GPU.

---

## Step 2: Configure Security Group

Create or modify a security group with these inbound rules:

| Type | Protocol | Port Range | Source | Purpose |
|------|----------|------------|--------|---------|
| SSH | TCP | 22 | Your IP | SSH access |
| Custom TCP | TCP | 5000 | 0.0.0.0/0 | Flask application |
| HTTP | TCP | 80 | 0.0.0.0/0 | (Optional) Nginx reverse proxy |
| HTTPS | TCP | 443 | 0.0.0.0/0 | (Optional) SSL |

---

## Step 3: Connect to EC2

```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

---

## Step 4: Install Docker

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo apt-get install -y docker-compose-plugin

# Add user to docker group (no sudo needed for docker commands)
sudo usermod -aG docker ubuntu
newgrp docker

# Verify installation
docker --version
docker compose version
```

---

## Step 5: Deploy Application

### Option A: Clone from Git

```bash
# Clone repository
git clone https://github.com/your-username/PlacementMentorAI.git
cd PlacementMentorAI
```

### Option B: Transfer files via SCP

```bash
# From your local machine
scp -i "your-key.pem" -r ./PlacementMentorAI ubuntu@<ec2-ip>:~/
```

---

## Step 6: Build and Run

### Using Docker Compose (Recommended)

```bash
cd PlacementMentorAI

# Build the image
docker compose build

# Run in detached mode
docker compose up -d

# View logs
docker compose logs -f
```

### Using Docker Directly

```bash
# Build
docker build -t placementmentorai .

# Run
docker run -d \
  --name placementmentorai \
  -p 5000:5000 \
  --shm-size=2g \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/cv_module/outputs:/app/cv_module/outputs \
  -v $(pwd)/cv_module/models:/app/cv_module/models \
  --restart unless-stopped \
  placementmentorai
```

---

## Step 7: Access Application

Open your browser and navigate to:

```
http://<your-ec2-public-ip>:5000
```

### Pages
| URL | Purpose |
|-----|---------|
| `http://<ip>:5000/` | Placement Score Prediction |
| `http://<ip>:5000/interview` | Interview Behaviour Analysis |

---

## Step 8: (Optional) Set Up Nginx Reverse Proxy

For production, use Nginx to proxy port 80 → 5000:

```bash
sudo apt-get install -y nginx

sudo tee /etc/nginx/sites-available/placementmentorai << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/placementmentorai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

---

## Management Commands

```bash
# View running containers
docker ps

# View logs
docker compose logs -f

# Restart the service
docker compose restart

# Stop the service
docker compose down

# Rebuild after code changes
docker compose up -d --build

# Check container resource usage
docker stats placementmentorai
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Container exits immediately | Check logs: `docker compose logs` |
| Out of memory | Use a larger instance or reduce workers in Dockerfile |
| Port 5000 not accessible | Verify Security Group allows TCP 5000 inbound |
| Video analysis timeout | Increase `--timeout` in Dockerfile CMD |
| Model download fails | Ensure outbound internet access from EC2 |
