# Changes to Evaluation Protocols and Fixes

* Minor fix where the examples with ground-truth mask all background are discarded in the "When" analysis, Sep 2025.
* LLaVA-G evaluation on PixMMVP was fixed, Jan 2026.
* Added the SpaCy similarity filtration for noun phrases below 0.7 following their implementation on PixMMVP, Jan 2026.
* Pixfoundation results were updated to use GPT-5.1 and performing automatic filtration w GPT 5.1 in PixMMVP for images that do not include the queried expression.
* PixFoundation (oracle) results were updated to include filtration of the images that do not include the queried expression using the groundtruth mask.

