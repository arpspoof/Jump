import argparse

allowed_contacts = {
                "walk":     "walk.txt",
                "cartwheel":"cartwheel.txt",
                "crawl":    "crawl.txt",
                "roll":     "roll.txt",
                "knee":     "knee.txt",
                }

def load_contact_file(contact_type):
    contact_parser = argparse.ArgumentParser()
    contact_parser.add_argument("N", type=int, nargs="+", help="allowed contact body ids")
    contact_file = "data/contacts/%s" % allowed_contacts[contact_type]
    with open(contact_file) as fh:
        s = fh.read().split()
        args = contact_parser.parse_args(s)
        allowed_body_ids = args.N
    return allowed_body_ids
