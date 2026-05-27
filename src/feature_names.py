from typing import List

FEATURE_NAME_MAP = {
    "sport": "Source port",
    "dsport": "Destination port",
    "dur": "Connection duration",
    "sbytes": "Source bytes",
    "dbytes": "Destination bytes",
    "sttl": "Source TTL",
    "dttl": "Destination TTL",
    "sloss": "Source packet loss",
    "dloss": "Destination packet loss",
    "sload": "Source load",
    "dload": "Destination load",
    "spkts": "Source packets",
    "dpkts": "Destination packets",
    "swin": "Source window size",
    "dwin": "Destination window size",
    "stcpb": "Source TCP bytes",
    "dtcpb": "Destination TCP bytes",
    "smeansz": "Source mean packet size",
    "dmeansz": "Destination mean packet size",
    "trans_depth": "HTTP transaction depth",
    "res_bdy_len": "Response body length",
    "sjit": "Source jitter",
    "djit": "Destination jitter",
    "sintpkt": "Source inter-packet time",
    "dintpkt": "Destination inter-packet time",
    "tcprtt": "TCP RTT",
    "synack": "SYN-ACK response time",
    "ackdat": "ACK data",
    "is_sm_ips_ports": "Suspicious IP/port indicator",
    "ct_state_ttl": "Count of state-TTL flow pairs",
    "ct_flw_http_mthd": "HTTP method count",
    "is_ftp_login": "FTP login indicator",
    "ct_ftp_cmd": "FTP command count",
    "ct_srv_src": "Count of same-service source flows",
    "ct_srv_dst": "Count of same-service destination flows",
    "ct_dst_ltm": "Count of flows to same dest in last minute",
    "ct_src_ltm": "Count of flows from same source in last minute",
    "ct_src_dport_ltm": "Count of flows from same source to same dest port in last minute",
    "ct_dst_sport_ltm": "Count of flows to same dest from same source port in last minute",
    "ct_dst_src_ltm": "Count of flows to same dest from same source in last minute",
}

OHE_PARENT_NAME_MAP = {
    "proto": "Protocol",
    "state": "Connection state",
    "service": "Service",
}


def humanize_feature_name(name: str) -> str:
    if "=" in name:
        feat, value = name.split("=", 1)
        parent = OHE_PARENT_NAME_MAP.get(feat, feat)
        return f"{parent}={value}"
    return FEATURE_NAME_MAP.get(name, name)


def humanize_feature_names(names: List[str]) -> List[str]:
    return [humanize_feature_name(name) for name in names]
