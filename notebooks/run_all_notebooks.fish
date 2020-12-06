#!/usr/bin/fish


for nb in *.ipynb
  echo $nb
  # and jupytext --to py:percent $nb
  jupyter nbconvert --execute --to notebook --inplace $nb
  echo
end
