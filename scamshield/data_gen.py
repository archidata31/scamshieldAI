"""Synthetic dataset generator for ScamShield.

No real dataset can be downloaded in every environment, so this builds a
labelled corpus of Indian-context SMS / email messages (1 = scam, 0 = legit).

Templates for every category are split into a "train" pool and a "test" pool,
so the test set contains phrasings the model has never seen. That makes the
reported metrics a fairer (still optimistic) estimate than a random split.

To use real data (SMS Spam Collection, Phishing Email dataset, ...) pass it to
train.py with --extra-csv.
"""
import argparse
import csv
import random
import string
from pathlib import Path

BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Kotak", "Bank of Baroda",
         "Canara Bank", "Paytm Payments Bank", "IDFC First"]
BRANDS = ["Amazon", "Flipkart", "Google", "Microsoft", "Netflix", "Jio", "Airtel",
          "Paytm", "PhonePe", "WhatsApp"]
MERCHANTS = ["Swiggy", "Zomato", "Amazon", "Flipkart", "BigBasket", "Myntra",
             "Uber", "IRCTC", "DMart", "Reliance Retail"]
NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Karan", "Pooja",
         "Rohit", "Neha", "Arjun", "Divya", "Suresh", "Meera"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
HANDLES = ["oksbi", "okhdfcbank", "ybl", "paytm", "axl", "ibl"]
SAFE_URLS = [
    "https://www.amazon.in/gp/your-orders", "https://www.flipkart.com/account/orders",
    "https://www.hdfcbank.com/", "https://www.onlinesbi.sbi", "https://www.swiggy.com/track",
    "https://www.irctc.co.in/nget/train-search", "https://www.icicibank.com/",
    "https://www.zomato.com/orders", "https://www.myntra.com/my/orders",
]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "t.ly", "rb.gy", "is.gd"]
URL_WORDS = ["kyc", "verify", "secure", "update", "login", "claim", "refund", "reward", "unlock"]
BAD_TLDS = ["xyz", "top", "click", "online", "site", "icu", "live", "vip", "cc", "in", "co"]


def _rand(n, chars=string.ascii_letters + string.digits):
    return "".join(random.choice(chars) for _ in range(n))


def scam_url():
    kind = random.random()
    if kind < 0.4:
        return f"{random.choice(['http://', 'https://', ''])}{random.choice(SHORTENERS)}/{_rand(random.randint(5, 7))}"
    if kind < 0.9:
        brand = random.choice(BANKS + BRANDS).lower().replace(" ", "")
        return (f"{random.choice(['http://', 'https://'])}{brand}-{random.choice(URL_WORDS)}"
                f".{random.choice(BAD_TLDS)}/{_rand(random.randint(4, 8))}")
    ip = ".".join(str(random.randint(11, 223)) for _ in range(4))
    return f"http://{ip}/{random.choice(URL_WORDS)}"


def phone():
    return f"{random.choice('6789')}{''.join(random.choice(string.digits) for _ in range(9))}"


def fill(template):
    amt = random.choice([499, 999, 1500, 2000, 5000, 9800, 12500, 25000, 50000, 100000, 250000])
    ctx = dict(
        bank=random.choice(BANKS), brand=random.choice(BRANDS), merchant=random.choice(MERCHANTS),
        name=random.choice(NAMES), month=random.choice(MONTHS),
        amt=f"{amt:,}", amt2=f"{random.randint(1000, 90000):,}",
        acct=str(random.randint(1000, 9999)), otp=str(random.randint(100000, 999999)),
        ref=str(random.randint(10**8, 10**10)),
        date=f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-2026",
        url=scam_url(), safeurl=random.choice(SAFE_URLS), phone=phone(),
        upi=f"{random.choice(NAMES).lower()}{random.randint(10, 99)}@{random.choice(HANDLES)}",
    )
    return template.format(**ctx)


# --------------------------------------------------------------------------
# SCAM templates: category -> (train templates, test templates)
# --------------------------------------------------------------------------
SCAM = {
    "kyc_block": (
        ["Dear customer, your {bank} account will be blocked today. Update your KYC immediately: {url}",
         "URGENT: {bank} KYC expired. Complete verification within 24 hrs to avoid suspension {url}",
         "Your {bank} a/c has been temporarily suspended due to incomplete KYC. Click {url} to reactivate now",
         "Dear {bank} user, PAN card not updated. Your account will be closed. Update here {url}",
         "{bank} Alert: Your net banking is locked. Verify your details now at {url} or call {phone}",
         "Final notice! Re-KYC pending for your {bank} account. Account freeze in 2 hours. {url}"],
        ["Hello, ur {bank} account is on hold. Share Aadhaar and PAN via {url} to unfreeze",
         "Your debit card is deactivated. Confirm card number and CVV at {url} to continue using it",
         "{bank} security team: unusual login detected. Verify identity immediately {url}"]),
    "upi_collect": (
        ["You have received a collect request of Rs {amt} from {upi}. Enter your UPI PIN to receive money",
         "Congrats! Rs {amt} cashback credited. Approve request on PhonePe and enter PIN to claim",
         "Payment of Rs {amt} pending. Scan this QR and enter UPI PIN to receive refund {url}",
         "Your refund of Rs {amt} is ready. Accept the collect request and share OTP to receive it",
         "Sir I sent Rs {amt} by mistake to your number. Please return urgently, here is my UPI {upi}"],
        ["Claim your Rs {amt} refund: open GPay, approve the request and type your PIN {url}",
         "Wrong transfer of Rs {amt}! Kindly send back to {upi} immediately, I am in emergency",
         "Scan the QR code to receive Rs {amt} prize amount. Enter PIN to confirm"]),
    "lottery_prize": (
        ["Congratulations! You have won Rs {amt} in the {brand} lucky draw. Claim now: {url}",
         "WINNER! You are selected for a free iPhone. Pay Rs 499 processing fee to claim {url}",
         "You won a KBC lottery of Rs {amt}. Contact {phone} with your bank details to claim prize",
         "Free gift voucher worth Rs {amt} waiting for you! Limited time. Click {url}"],
        ["Lucky customer! Your mobile number won Rs {amt}. Send registration fee to {upi} to receive",
         "You've been chosen for exclusive reward of Rs {amt}. Offer expires tonight {url}"]),
    "delivery_utility": (
        ["Your package is held at customs. Pay Rs {amt} clearance fee at {url} to release",
         "India Post: delivery failed due to incorrect address. Reschedule and pay Rs 25 at {url}",
         "Dear consumer, your electricity will be disconnected tonight 9:30 PM due to unpaid bill. Call {phone} immediately",
         "Your FASTag is blocked. Recharge now {url} to avoid penalty",
         "Your SIM will be deactivated in 2 hours. Complete verification {url}"],
        ["Courier unable to deliver your parcel. Confirm address and pay Rs 30 fee: {url}",
         "Your gas connection will be cut. Pay pending bill urgently via {url}",
         "TRAI notice: your mobile number will be disconnected. Press 1 or call {phone}"]),
    "job_invest_loan": (
        ["Work from home! Earn Rs {amt} per day by liking videos. Join our Telegram {url}",
         "Part-time job offer. Earn Rs {amt} daily. WhatsApp {phone} now, limited slots",
         "Guaranteed 300% returns in 7 days. Join our VIP stock tips group {url}",
         "Instant loan Rs {amt} approved without documents. Pay processing fee Rs 999 to {upi}",
         "Invest Rs 5000 and earn Rs 50000 in crypto. Double profit guaranteed {url}"],
        ["HR here: you are shortlisted. Earn Rs {amt}/day from home, no experience. Register {url}",
         "Pre-approved loan of Rs {amt}! No CIBIL check. Pay small fee to {upi} for disbursal",
         "Sure shot profit tips, 100% accurate. Join paid group today {url}"]),
    "otp_phishing": (
        ["Your {bank} account has been compromised. Share the OTP you received to secure it",
         "Dear user, we noticed suspicious activity on your {brand} account. Sign in at {url} to confirm your password",
         "Subject: Action required - verify your password. Your mailbox will expire today. Login {url}",
         "This is {bank} customer care. For card upgrade, please share the 6 digit OTP",
         "Income Tax Dept: refund of Rs {amt} approved. Submit bank details at {url}"],
        ["Security alert from {brand}: unauthorized sign-in. Reset password now {url}",
         "Customer care here, to stop deduction please tell us the code sent to your phone",
         "Tax refund pending Rs {amt}. Verify account number and ATM PIN {url}"]),
}

# --------------------------------------------------------------------------
# LEGIT templates (incl. hard negatives: real bank alerts, OTPs, urgent work mail)
# --------------------------------------------------------------------------
HAM = {
    "bank_alert": (
        ["Rs {amt} debited from A/c XX{acct} on {date} to {name}. Avl bal Rs {amt2}. If not you, call bank helpline.",
         "Your OTP for transaction of Rs {amt} at {merchant} is {otp}. Do not share this OTP with anyone. -{bank}",
         "Rs {amt} credited to your account XX{acct} on {date} via UPI from {name}. Ref no {ref}.",
         "{bank}: Your statement for {month} is ready. Login to the official app to view it.",
         "Your credit card ending {acct} bill of Rs {amt} is due on {date}. Pay via the {bank} app. Ignore if already paid.",
         "UPI txn of Rs {amt} to {merchant} successful. UPI Ref {ref}."],
        ["Dear customer, Rs {amt} withdrawn at ATM on {date}. Bal Rs {amt2}. Never share your PIN or OTP with anyone.",
         "{bank} Alert: Rs {amt} spent on card XX{acct} at {merchant}. Not you? Call the number on back of card.",
         "Your SIP of Rs {amt} was processed on {date}. Units allotted. Ref {ref}."]),
    "personal": (
        ["hey are you coming for dinner tonight?", "Ok I will call you in 10 mins",
         "Happy birthday {name}! Have a great year ahead", "Mom said bring vegetables on your way home",
         "Can you send the notes from yesterday's class?", "Reached office. Will join the call shortly",
         "Bro I'll pay you back the Rs {amt} on Sunday, don't worry"],
        ["Lunch at 1? Let me know", "Did you submit the assignment?",
         "Thanks {name}, will transfer the rent on Monday", "Traffic is crazy, running 15 mins late"]),
    "work": (
        ["Hi team, the sprint review is scheduled for {date} at 4pm. Agenda attached.",
         "Urgent: please review the quarterly report before tomorrow's meeting.",
         "Your interview for the Data Analyst role is confirmed on {date}. Please bring your ID.",
         "Invoice {ref} for Rs {amt} has been generated. Please find it attached.",
         "Please confirm your attendance for the workshop by replying to this email."],
        ["Reminder: standup moved to 10:30 tomorrow", "Your leave request for {date} has been approved by your manager",
         "Please find attached the minutes of today's meeting. Let me know if anything is missing."]),
    "service": (
        ["Your {merchant} order {ref} is out for delivery today. Track: {safeurl}",
         "Flat 20% off on your next order with code SAVE20. Valid till {date}. T&C apply.",
         "Your {merchant} OTP is {otp}. Valid for 10 minutes. Do not share with anyone.",
         "Your electricity bill of Rs {amt} for {month} has been paid successfully. Thank you.",
         "Your {brand} verification code is {otp}. Never share this code with anyone."],
        ["Your parcel from {merchant} was delivered. Rate your experience at {safeurl}",
         "Appointment reminder: Dr. {name} tomorrow at 11 AM. Reply C to confirm",
         "Your recharge of Rs {amt} is successful. Validity 28 days."]),
}


SCAM["digital_arrest"] = (
    ["Cyber crime branch notice: a parcel in your name contains illegal items. Contact officer on {phone} immediately or face arrest",
     "Your Aadhaar is linked to a money laundering case. Call {phone} now to avoid arrest",
     "CBI notice: your SIM was used for illegal activity. Do not disconnect, join video call and verify identity at {url}"],
    ["This is TRAI. Your number is involved in illegal activity. Stay on video call, do not tell anyone, call {phone}",
     "Narcotics found in courier booked using your ID. Speak to the officer at {phone} urgently to avoid arrest"])
SCAM["hinglish"] = (
    ["Aapka {bank} account aaj band ho jayega. Turant KYC update karein: {url}",
     "Badhai ho! Aapne {amt} rupaye ka inaam jeeta hai. Claim karne ke liye {url} par click karein",
     "Apna OTP abhi batayein warna aapka account block ho jayega",
     "Aapka {bank} card block ho gaya hai. Unblock karne ke liye {phone} par turant call karein"],
    ["Sir aapka KYC pending hai, aaj hi update na kiya to account freeze ho jayega {url}",
     "Aapko {amt} ka cashback mila hai. UPI PIN daalkar paise receive karein"])
HAM["hinglish"] = (
    ["Bhai aaj raat dinner pe aa raha hai kya?", "Mummy ne bola sabzi leke aana",
     "Aapke account se Rs {amt} debit hue hain {date} ko. Agar aapne nahi kiya toh bank ko call karein.",
     "Kal subah 10 baje office pahunch jana, meeting hai"],
    ["Kal meeting 11 baje hai, time pe aa jana", "Aapka OTP {otp} hai. Kisi ke saath share na karein.",
     "Yaar paise Sunday ko bhej dunga, tension mat le"])


def noisy(text):
    """Mimic real SMS: dropped punctuation, lowercase, typos, filler words."""
    r = random.random()
    if r < 0.18:
        text = text.lower()
    elif r < 0.30:
        text = text.translate(str.maketrans("", "", ".,!:"))
    if random.random() < 0.10 and len(text) > 20:
        i = random.randint(3, len(text) - 3)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if random.random() < 0.08:
        text = text.replace("you", "u").replace("your", "ur")
    return text


def generate(n, split, scam_ratio=0.38, seed=0):
    random.seed(seed)
    idx = 0 if split == "train" else 1
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < n * 20:
        guard += 1
        is_scam = random.random() < scam_ratio
        pool = SCAM if is_scam else HAM
        cat = random.choice(list(pool))
        t_idx = random.randrange(len(pool[cat][idx]))
        text = noisy(fill(pool[cat][idx][t_idx]))
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": int(is_scam), "category": cat, "split": split,
                     "source": "synthetic", "group": f"{split}-{cat}-{int(is_scam)}-{t_idx}"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--out", default="data/synthetic.csv")
    a = ap.parse_args()
    rows = generate(a.train, "train", seed=1) + generate(a.test, "test", seed=2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "category", "split", "source", "group"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
