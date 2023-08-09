
###  Ubuntu 20.04 22.04 ssh

```rust

sudo vim /etc/ssh/sshd_config

HostKeyAlgorithms +ssh-rsa
PubkeyAcceptedKeyTypes +ssh-rsa

KexAlgorithms curve25519-sha256@libssh.org,ecdh-sha2-nistp256,ecdh-sha2-nistp384,ecdh-sha2-nistp521,diffie-hellman-group-exchange-sha256,diffie-hellman-group14-sha1,diffie-hellman-group-exchange-sha1,diffie-hellman-group1-sha1

```

