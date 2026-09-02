#!/usr/bin/env bash

# Source this file from a Slurm task before torchrun. The first argument must
# resolve to a peer compute node, not the current host.
peer_host="${1:?peer host is required}"
peer_ip="$(getent ahostsv4 "$peer_host" | awk '$2 == "STREAM" {print $1; exit}')"
[[ -n "$peer_ip" ]]

peer_route="$(ip -o -4 route get "$peer_ip")"
distributed_interface="$(
  awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}' \
    <<<"$peer_route"
)"
distributed_source_ip="$(
  awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}' \
    <<<"$peer_route"
)"

[[ -n "$distributed_interface" && "$distributed_interface" != "lo" ]]
[[ -n "$distributed_source_ip" ]]
[[ "$distributed_source_ip" != 127.* && "$distributed_source_ip" != 169.254.* ]]

export NCCL_SOCKET_IFNAME="$distributed_interface"
export GLOO_SOCKET_IFNAME="$distributed_interface"
export OPENPROTEIN_DISTRIBUTED_INTERFACE="$distributed_interface"
export OPENPROTEIN_DISTRIBUTED_SOURCE_IP="$distributed_source_ip"
export OPENPROTEIN_DISTRIBUTED_PEER_HOST="$peer_host"
export OPENPROTEIN_DISTRIBUTED_PEER_IP="$peer_ip"

printf 'distributed_route host=%s peer_host=%s peer_ip=%s interface=%s source_ip=%s\n' \
  "$(hostname)" "$peer_host" "$peer_ip" "$distributed_interface" \
  "$distributed_source_ip"
