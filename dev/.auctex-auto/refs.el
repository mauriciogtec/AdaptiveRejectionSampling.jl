;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "refs"
 (lambda ()
   (LaTeX-add-bibitems
    "Gilks_Wild_1992"
    "rasmussenInfiniteGaussianMixture"))
 '(or :bibtex :latex))

