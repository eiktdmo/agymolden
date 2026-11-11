# Private Google Site Setup Guide (Personal Google Account)

Follow these simple steps using your **current personal Google account** to publish a private website where:
1. Your university students (with their university emails) can access it.
2. Any other invited person can access it simply by adding their email.
3. Nobody else can see or download anything without your explicit permission.

---

## Step 1: Upload the One-Click Bundle to Google Drive
1. Open your browser and go to: **[drive.google.com](https://drive.google.com)** (using your current personal Google account).
2. Create a new folder named `AgyMolden Private Beta`.
3. Drag and drop the file:
   `AgyMolden_OneClick_Bundle.zip` (located in `/home/andres/Desktop/newagy/AgyMolden_OneClick_Bundle.zip`) into this folder.
4. Right-click the uploaded `.zip` file in Google Drive $\to$ **Share** $\to$ **Share**:
   - Set General access to: **Restricted** (Only people with access can open with the link).
   - Click **Copy link** (we will use this download link in Step 2).

---

## Step 2: Create Your Private Google Site
1. Open: **[sites.google.com](https://sites.google.com)** (in the same personal Google account).
2. Click **Blank site** (`+`).
3. Set Site Title: `AgyMolden — Quantum Chemistry Beta`.
4. Add the Content:
   - On the right panel, click **Embed** (`</>`) $\to$ choose **Embed code**.
   - Open our prepared file:
     `/home/andres/Desktop/newagy/docs/private_beta_site/index.html`
   - Copy its content and paste it into the Embed code box.
   - Click **Next** $\to$ **Insert**, then expand the embedded frame to fill the page width.
   *(Or alternatively: you can add a Button widget linking directly to your Google Drive link from Step 1).*

---

## Step 3: Configure Privacy & Restrict Access to Students & Invited Emails
This is the key step that guarantees complete privacy:

1. In the top-right toolbar of Google Sites, click the **Share with others** icon (the person with the `+` icon).
2. In the sharing dialog:
   - Under **General access**: Ensure **Published site** is set to **Restricted**.
     *(When Restricted, only people you add can view the published site).*
3. Under **Add people and groups**:
   - Paste the email addresses of your students (e.g. `student@university.edu`, `student2@gmail.com`).
   - Paste any colleague or external tester emails you want to invite.
   - Make sure their role is set to **Viewer** (not Editor).
   - Check "Notify people" if you want Google to automatically send them an email invitation.
4. Click **Done**.
5. Click the purple **Publish** button in the top-right corner.
   - Choose a web address suffix (e.g., `agymolden-beta`).
   - Click **Publish**.

---

## Step 4: Adding New Students or People Later
Whenever you want to grant access to another student or colleague:
1. Open your Google Site in edit mode.
2. Click the **Share** (`+`) icon.
3. Type their email address and click **Done**.
4. They will immediately have access using their Google identity.

Anyone whose email is not on your list who attempts to open the link will see Google's:
> **"You need access — Ask for access, or switch to an account with access."**
