#!/usr/bin/fish


for nb in *.ipynb
  echo $nb
  jupyter-nbconvert --execute --clear-output $nb
  echo
end
