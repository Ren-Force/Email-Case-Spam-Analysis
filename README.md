# Spam Email Data Analysis
## Background
The client had seen a gradual increase in the number of spam cases as the business grew and product lines expanded, which required the support team to manually open each case, read the original email, and mark it as spam. This process was time-consuming and inefficient. **Additionally, spam emails are very likely fraud-related, so preventing spam emails from being exposed to the user would help reduce the risk of fraud.** My goal was to improve this workflow by analyzing existing spam data and identifying ways to potentially automate spam detection.

## Purpose of Analysis
To help the support team focus on real customer cases and reduce distractions, I conducted a spam analysis. This analysis aimed to:
- Prevent future spam cases from entering Salesforce by providing a list of spam email addresses/domains.
- Analyze email subjects and bodies to offer insights into commonly marked spam content and the sentiment of these emails.
- Identify potential patterns of phrases commonly used in spam emails, to assist in flagging them more efficiently in the future.

## Scope of Analysis
- ### Cases Generated from Email:
  - I focused on cases created from email communications.
- ### Initial Incoming Email for Analysis:
  - Only the first email received in each case was analyzed.
- ### Auto-Response Addresses Filtered:
  - Auto-response emails were excluded to ensure only relevant data was considered.

## Data Preparation
To prepare the email messages for analysis, these steps were taken to extract, clean, and categorize the data:
- ### Extract Email Messaging Data from Salesforce:
  - Exported Email Message ID, Subject, Text Body, Parent Case Status, From Address, and Created Date, excluding internal company emails.
- ### Clean Email Messaging Data:
  - Excluded auto-responses such as “Out-of-office” and “Not Delivered.” and the origin email addresses
- ### Group Email Messages:
  - Split email messages into "Junk & Spam" and "Non-Junk & Spam" groups for comparison.
 
## Exploratory Data Analysis
- Analyzed Subject and TextBody fields to identify common patterns, words, or phrases frequently associated with spam.
- Investigated FromAddress to detect patterns of known spam email addresses or domains.
- Analyzed the CreatedDate to understand spam frequency and timing.
- Used the Parent.Status field to track how cases were being resolved and whether they were flagged as spam.

## NLP for Spam Detection and Pattern Identification
Out of curiosity, I tried using NLP and pattern identification to see if I could discover any specific patterns in spam emails. By tokenizing and vectorizing the TextBody field, I explored word frequency and looked for recurring phrases. However, the results were not particularly effective in accurately identifying spam emails.

While this approach didn’t significantly improve spam detection, it may still be useful in different contexts or use cases where specific phrase detection could be valuable.

## Future Improvement
If I were to continue exploring the likelihood of spam based on email cases, I could employ more advanced techniques like regression models or NNMs. These methods could help identify which phrases and patterns are most likely to affect the probability of an email being spam.

That said, since the deliverable needed to align with the client’s budget and my work capacity, I didn’t pursue this direction further. If you come across similar requests or think this might be helpful for your own use case, feel free to add pull requests or leave your comments!

