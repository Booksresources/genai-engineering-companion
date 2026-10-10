def monthly_cost(requests_per_day, in_tokens, out_tokens, price_in_per_m, price_out_per_m, days=30):
    """Estimate monthly API spend from traffic and per-million-token prices."""
    per_request = (in_tokens * price_in_per_m + out_tokens * price_out_per_m) / 1_000_000
    return requests_per_day * days * per_request

for name, pin, pout in [("small", 0.10, 0.40), ("mid-size", 1.0, 4.0), ("large", 5.0, 15.0)]:
    print(f"{name:9s} 10,000 req/day -> ${monthly_cost(10_000, 1500, 300, pin, pout):>9,.0f} per month")
