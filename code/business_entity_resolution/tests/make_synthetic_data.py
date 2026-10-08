"""
make_synthetic_data.py
FOR LOCAL PIPELINE TESTING ONLY. This generates made-up business names and
addresses purely to check that blocking.py / features.py / train.py /
predict.py run end-to-end without errors. It is NOT real competition data
and should not be used to tune the actual model -- swap in the real
dataset/train and dataset/test files for that.
"""
import random
import string

random.seed(0)

NAME_ROOTS = ["Acme", "Bluepeak", "Sunrise", "Oakridge", "Silverline", "Zenith",
              "Riverdale", "Northstar", "Coral", "Meridian", "Highfield", "Vertex",
              "Falcon", "Ember", "Granite", "Willowmere", "Cobalt", "Ashgrove",
              "Ironwood", "Pinnacle", "Cedarfield", "Marigold", "Stonebridge",
              "Lumen", "Harborview", "Redwood", "Quartz", "Brightwater",
              "Frostline", "Elmcourt", "Sablewood", "Junction", "Cascade",
              "Milestone", "Thistledown", "Hollowbrook", "Amberline", "Cypress",
              "Driftwood", "Wrenfield"]
NAME_TYPES_US = ["Corp", "Inc", "LLC", "Co"]
NAME_TYPES_IN = ["Pvt Ltd", "Ltd", "Enterprises"]
STREETS = ["Main St", "5th Ave", "Park Rd", "MG Road", "Church St", "Oak Lane", "Ring Rd"]
CITIES_US = ["Springfield", "Fairview", "Greenville", "Madison"]
CITIES_IN = ["Bengaluru", "Pune", "Chennai", "Hyderabad"]


def rand_postal(country):
    if country == "US":
        return "".join(random.choices(string.digits, k=5))
    return "".join(random.choices(string.digits, k=6))


def make_base(i, country):
    root = random.choice(NAME_ROOTS)
    suffix = random.choice(NAME_TYPES_US if country == "US" else NAME_TYPES_IN)
    name = f"{root} {suffix}"
    street = random.choice(STREETS)
    city = random.choice(CITIES_US if country == "US" else CITIES_IN)
    num = random.randint(1, 999)
    addr = f"{num} {street}, {city}, {country} {rand_postal(country)}"
    return name, addr


def noisy_name(name):
    repl = {"Corp": "Corporation", "Inc": "Incorporated", "LLC": "Limited Liability Company",
            "Pvt Ltd": "Private Limited", "Ltd": "Limited"}
    for k, v in repl.items():
        if k in name and random.random() < 0.5:
            name = name.replace(k, v)
    if random.random() < 0.2:
        name = name.replace(" ", "  ")  # extra whitespace noise
    return name

def noisy_addr(addr):
    if random.random() < 0.3:
        addr = "Near City Mall, " + addr
    if random.random() < 0.3:
        addr = addr.replace("Road", "Rd").replace("Street", "St")
    if random.random() < 0.2:
        addr = addr.rsplit(",", 1)[0]  # drop postal code
    return addr


def build(n_entities=120, match_rate=0.7, out_dir="dataset"):
    import os
    os.makedirs(f"{out_dir}/train", exist_ok=True)
    os.makedirs(f"{out_dir}/test", exist_ok=True)

    def gen_split(prefix, n, start_id=1):
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        eid = start_id
        for i in range(n):
            country = "US" if random.random() < 0.5 else "India"
            base_country = "US" if country == "US" else "IN"
            name, addr = make_base(i, base_country)
            s1_id = f"S1-{eid:05d}"
            s1_rows.append((s1_id, name, addr, country))

            matched_ids = []
            if random.random() < match_rate:
                if random.random() < 0.7:
                    mid = f"S2-{eid:05d}"
                    s2_rows.append((mid, noisy_name(name), noisy_addr(addr), country))
                    matched_ids.append(mid)
                if random.random() < 0.5:
                    mid3 = f"S3-{eid:05d}"
                    s3_rows.append((mid3, noisy_name(name), noisy_addr(addr), country))
                    matched_ids.append(mid3)
            gt_rows.append((s1_id, ",".join(matched_ids)))
            eid += 1

        # add some pure-noise (non-matching) S2/S3 records to force the
        # blocker/model to actually discriminate
        for j in range(n // 2):
            country = "US" if random.random() < 0.5 else "India"
            base_country = "US" if country == "US" else "IN"
            name, addr = make_base(1000 + j, base_country)
            if random.random() < 0.5:
                s2_rows.append((f"S2-N{j:05d}", name, addr, country))
            else:
                s3_rows.append((f"S3-N{j:05d}", name, addr, country))

        return s1_rows, s2_rows, s3_rows, gt_rows

    def write_tsv(path, rows, cols):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")

    cols = ["entity_id", "business_name", "business_address", "country"]

    s1, s2, s3, gt = gen_split("train", n_entities, start_id=1)
    write_tsv(f"{out_dir}/train/train_source1.tsv", s1, cols)
    write_tsv(f"{out_dir}/train/train_source2.tsv", s2, cols)
    write_tsv(f"{out_dir}/train/train_source3.tsv", s3, cols)
    write_tsv(f"{out_dir}/train/train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])

    s1t, s2t, s3t, gtt = gen_split("test", n_entities // 2, start_id=9001)
    write_tsv(f"{out_dir}/test/test_source1.tsv", s1t, cols)
    write_tsv(f"{out_dir}/test/test_source2.tsv", s2t, cols)
    write_tsv(f"{out_dir}/test/test_source3.tsv", s3t, cols)
    # (test ground truth withheld in the real challenge; we don't write one)

    print("Synthetic data written to", out_dir)


if __name__ == "__main__":
    build()
