---
title: "Computer Network - Network Layer"
date:   2023-09-11
last_modified_at: 2020-10-06
categories: notes
tags: [Computer Network ]


---

# The Network Layer

## 1. Functional Overview

* **Datagram Service:**
* Simple, flexible, and connectionless.
* Best-effort delivery: The network layer does not guarantee reliability (packets may be dropped, duplicated, delayed, or arrive out of order); these functions are offloaded to the Transport Layer.


* **Heterogeneous Network Interconnection:**
* Layer 1: Repeaters/Hubs
* Layer 2: Bridges/Switches
* Layer 3: **Routers**
* Above Layer 3: Gateways


* **Routing vs. Forwarding:**
* **Routing (Control Plane):** Uses distributed algorithms to determine paths based on network topology.
* **Forwarding (Data Plane):** Uses a forwarding table to move IP datagrams to the appropriate output port.


* **SDN (Software Defined Networking):** Separates the data plane (forwarding) from the control plane (centralized routing logic).
* **Congestion Control:** Managed via open-loop and closed-loop control mechanisms.

---

## 2. Routing Algorithms

* **Static Routing:** Non-adaptive; configured manually by network administrators. Best for small networks.
* **Dynamic Routing:** Adaptive; routers exchange routing tables dynamically.
* **Distance-Vector Algorithm (e.g., RIP):** Nodes periodically broadcast their entire routing table to neighbors.
* **Link-State Algorithm (e.g., OSPF):** Each node maintains a complete network topology, tests neighbors, and floods link-state information when changes occur.
* **Hierarchical Routing:** The internet is divided into smaller **Autonomous Systems (AS)** to prevent massive, unmanageable routing tables.
* **IGP (Interior Gateway Protocol):** RIP, OSPF
* **EGP (Exterior Gateway Protocol):** BGP



---

## 3. IPv4 Protocol

### IPv4 Packet Header

* **Header Length:** 4 bits (Max: 15 $\times$ 4B = 60B).
* **Total Length:** Max $2^{16} = 65,535$ bytes.
* **Fragmentation Fields:**
* **Identification:** Shared by fragments of the same original packet.
* **Flags:**
* **MF (More Fragments):** 1 = more fragments follow; 0 = last fragment.
* **DF (Don't Fragment):** 1 = forbidden to fragment.


* **Fragment Offset:** 13 bits; indicates position in the original packet (units of 8 bytes).


* **TTL (Time to Live):** Prevents routing loops by limiting the number of routers a packet can traverse.
* **Protocol:** Defines the encapsulated transport protocol (e.g., TCP: 6, UDP: 17).
* **Header Checksum:** Validates the integrity of the header only.

### IP Addressing & Subnetting

* **Classful Addressing:** Classes A, B, C, D (Multicast), and E (Experimental).
* **NAT (Network Address Translation):** Maps private LAN IPs (Class A: 10.x.x.x, B: 172.16.x.x-172.31.x.x, C: 192.168.x.x) to public WAN IPs.
* **Subnetting & Subnet Mask:** Borrowing host bits to create subnets.
* **CIDR (Classless Inter-Domain Routing):** Replaces classful addressing using prefix length (e.g., `/20`).
* **Longest Prefix Matching:** When multiple routes match, the most specific one (longest prefix) is chosen.



### ARP, DHCP, and ICMP

* **ARP (Address Resolution Protocol):** Resolves IP addresses to MAC addresses for local network delivery.
* **DHCP (Dynamic Host Configuration Protocol):** Dynamically assigns IP addresses (Application Layer protocol using UDP).
* **ICMP (Internet Control Message Protocol):** Reports network errors and diagnostics (used by `ping` and `traceroute`).

---

## 4. IPv6

* **Features:** 128-bit address space, simplified header, no transit fragmentation (only source fragments), and automatic configuration.
* **Address Types:** Unicast, Multicast, and Anycast (delivered to the "nearest" node in a group).
* **Transition Technologies:** Dual-stack devices and tunneling (encapsulating IPv6 packets within IPv4).

---

## 5. Routing Protocols

* **RIP:** Distance-Vector, hop count limit of 15, uses UDP 520.
* **OSPF:** Link-State, uses Dijkstra’s algorithm for shortest path, fast convergence, uses IP protocol 89.
* **BGP:** Path-Vector, used between ASs, runs on TCP.

---

## 6. IP Multicast

* Uses D-class addresses (224.0.0.0 – 239.255.255.255).
* **IGMP (Internet Group Management Protocol):** Manages multicast group membership.

---

## 7. Mobile IP

* Allows roaming between networks while maintaining a permanent IP address.
* **Entities:** Mobile Node, Home Agent, and Foreign Agent.
* **Mechanism:** The Home Agent creates a tunnel to the Mobile Node’s "care-of" address in the visited network.

---

## 8. Network Layer Equipment

* **Collision Domain:** Nodes competing for the same medium (Hubs/Repeaters do not divide this; Switches/Routers do).
* **Broadcast Domain:** Nodes receiving the same broadcast (Only Routers divide this).