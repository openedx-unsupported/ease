# -*- mode: ruby -*-
# vi: set ft=ruby :

# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

$script = <<END
set -e

echo "Updating apt packages..."
apt-get update -y

echo "Installing basic system requirements..."
apt-get install -y curl git vim libxml2-dev libxslt1-dev memcached nginx

echo "Installing mysql server..."
DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server-5.5
echo "CREATE DATABASE IF NOT EXISTS workbench DEFAULT CHARACTER SET utf8 DEFAULT COLLATE utf8_general_ci;" | mysql -u root

echo "Installing Python system requirements..."
apt-get install -y python2.7 python2.7-dev python-pip python-software-properties python-mysqldb libmysqlclient-dev
pip install virtualenv

echo "Installing FireFox and xvfb (for JavaScript tests)..."
add-apt-repository "ppa:ubuntu-mozilla-security/ppa"
apt-get install -y firefox dbus-x11 xvfb

cat /home/vagrant/edx-ora2/vagrant/xvfb.conf > /etc/init/xvfb.conf
start xvfb || true

echo "Installing RabbitMQ..."
add-apt-repository "deb http://www.rabbitmq.com/debian/ testing main"
cd /tmp && wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc && apt-key add rabbitmq-signing-key-public.asc
apt-get update -y
apt-get install -y rabbitmq-server

echo "Installing NodeJS..."
add-apt-repository ppa:chris-lea/node.js
apt-get update -y
apt-get install -y nodejs

# Stop all Python upstart jobs
sudo stop workbench || true
sudo stop celery || true
sudo stop flower || true

su vagrant <<EOF
    set -e

    echo "Creating a virtualenv..."
    mkdir -p /home/vagrant/.virtualenvs
    virtualenv /home/vagrant/.virtualenvs/edx-ora2
    source /home/vagrant/.virtualenvs/edx-ora2/bin/activate

    echo "Configuring login script..."
    cat /home/vagrant/edx-ora2/vagrant/bash_profile > /home/vagrant/.bash_profile

    echo "Installing EASE..."
    if [ ! -d /home/vagrant/ease ]; then
        git clone https://github.com/edx/ease.git /home/vagrant/ease
    fi
    cat /home/vagrant/ease/apt-packages.txt | xargs sudo apt-get -y install
    cd /home/vagrant/ease && pip install -r pre-requirements.txt
    cd /home/vagrant/ease && python setup.py install

    echo "Downloading NLTK corpus..."
    cd /home/vagrant/ease && ./download-nltk-corpus.sh

    echo "Installing gunicorn..."
    pip install gunicorn

    echo "Instally Python MySQL library..."
    pip install MySQL-python

    echo "Installing celery flower..."
    pip install flower

    echo "Install edx-ora2..."
    cd /home/vagrant/edx-ora2 && ./scripts/install.sh

    echo "Update the database..."
    cd /home/vagrant/edx-ora2 && python manage.py syncdb --migrate --noinput --settings settings.vagrant

    echo "Collect static assets..."
    mkdir -p /home/vagrant/static
    cd /home/vagrant/edx-ora2 && python manage.py collectstatic --noinput --settings settings.vagrant

    echo "Creating the update script..."
    cp /home/vagrant/edx-ora2/vagrant/update.sh /home/vagrant/update.sh

EOF

echo "Creating upstart script for workbench..."
cat /home/vagrant/edx-ora2/vagrant/workbench_upstart.conf > /etc/init/workbench.conf
start workbench || true

echo "Create upstart script for Celery workers..."
cat /home/vagrant/edx-ora2/vagrant/celery_upstart.conf > /etc/init/celery.conf
start celery || true

echo "Create upstart script for Celery flower..."
cat /home/vagrant/edx-ora2/vagrant/flower_upstart.conf > /etc/init/flower.conf
start flower || true

echo "Configure nginx"
cat /home/vagrant/edx-ora2/vagrant/nginx.conf > /etc/nginx/sites-enabled/workbench.conf

echo "Restart nginx"
sudo service nginx stop || true
sudo service nginx start || true

END

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

  config.vm.box = "precise64"
  config.vm.box_url = "http://files.vagrantup.com/precise64.box"

  config.vm.network "private_network", ip: "192.168.44.10"
  config.vm.synced_folder ".", "/home/vagrant/edx-ora2"


  config.vm.provider :virtualbox do |vb|
    # Increase memory and CPU
    vb.customize ["modifyvm", :id, "--memory", "2048"]
    vb.customize ["modifyvm", :id, "--cpus", "2"]

    # Allow DNS to work for Ubuntu 12.10 host
    # http://askubuntu.com/questions/238040/how-do-i-fix-name-service-for-vagrant-client
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  config.vm.provision "shell", inline: $script
end
