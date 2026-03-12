#!/bin/bash
set -e

VERSION_FILE="VERSION"

if [ ! -f "$VERSION_FILE" ]; then
  echo "ERROR: No se encuentra $VERSION_FILE"
  exit 1
fi

# Verificar que no hay cambios sin commitear
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo "ERROR: Hay cambios sin commitear. Haz commit o stash antes de hacer release."
  exit 1
fi

CURRENT=$(cat "$VERSION_FILE" | tr -d '[:space:]')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

echo "Version actual: $CURRENT"
echo ""
echo "Tipo de release:"
echo "  1) patch  ($MAJOR.$MINOR.$((PATCH + 1)))"
echo "  2) minor  ($MAJOR.$((MINOR + 1)).0)"
echo "  3) major  ($(( MAJOR + 1)).0.0)"
read -p "Selecciona (1/2/3): " choice

case $choice in
  1) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
  2) NEW="$MAJOR.$((MINOR + 1)).0" ;;
  3) NEW="$((MAJOR + 1)).0.0" ;;
  *) echo "Opcion no valida" && exit 1 ;;
esac

echo "$NEW" > "$VERSION_FILE"

git add "$VERSION_FILE"
git commit -m "release: wkg-v$NEW"
git tag "wkg-v$NEW"

echo ""
echo "Tag creado: wkg-v$NEW"
echo ""
echo "Ejecuta para desplegar:"
echo "   git push && git push --tags"
