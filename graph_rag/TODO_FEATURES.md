
# Graph-RAG: Funcionalidades Implementadas y Pendientes

Este documento contiene el estado de todas las funcionalidades del sistema.

## Estado Actual

El sistema tiene:
- ✅ Supabase con pgvector funcionando
- ✅ Neo4j para grafo de enlaces
- ✅ Embeddings con multilingual-e5-large
- ✅ RAG básico funcionando (vector + graph expansion)
- ✅ Dashboard HTML conectado
- ✅ Crawler y servicio de ingest
- ✅ Interlinking suggestions
- ✅ **PageRank/HITS calculado automáticamente**
- ✅ **Historial de conversaciones**
- ✅ **Gestión completa de clientes**

## Funcionalidades IMPLEMENTADAS

### ✅ 1. Cálculo de PageRank/HITS en Neo4j
- **Ubicación**: `neo4j_client.py:363-593`
- **Métodos**:
  - `calculate_pagerank()` - Algoritmo iterativo con damping factor
  - `calculate_hits()` - Hub y Authority scores
  - `calculate_all_scores()` - Ejecuta ambos
  - `get_top_pages_by_pagerank()` - Top páginas ordenadas
- **Endpoint**: `POST /api/v1/dashboard/clients/{id}/calculate-scores`

### ✅ 2. Historial de Conversaciones en Chat
- **Ubicación**: `routes.py:352-431`, `supabase_client.py:405-493`
- **Endpoints**:
  - `GET /api/v1/dashboard/conversations/{client_id}` - Lista conversaciones
  - `GET /api/v1/dashboard/conversation/{id}` - Ver mensajes
  - `DELETE /api/v1/dashboard/conversation/{id}` - Eliminar
  - `POST /api/v1/dashboard/conversation/new` - Crear nueva
- **El chat guarda automáticamente mensajes** con session_id/conversation_id

### ✅ 3. Recálculo de Scores Post-Ingest
- **Ubicación**: `ingest_service.py:207-220`
- **Comportamiento**: Después de ingestar links, calcula PageRank/HITS automáticamente
- **Sincroniza** scores a Supabase

### ✅ 4. Eliminar Cliente y sus Datos
- **Ubicación**: `supabase_client.py:559-593`, `routes.py:952-977`
- **Endpoints**:
  - `DELETE /api/v1/dashboard/clients/{id}` - Eliminar completamente
  - `POST /api/v1/dashboard/clients/{id}/deactivate` - Soft delete
- **Elimina en cascada**: messages → conversations → links → pages → client

### ✅ 5. Regenerar Embeddings
- **Ubicación**: `ingest_service.py:239-302`, `routes.py:1019-1047`
- **Endpoint**: `POST /api/v1/dashboard/clients/{id}/regenerate-embeddings`
- **Itera** sobre todas las páginas con contenido y regenera embeddings

### ✅ 6. Rotación de API Key
- **Ubicación**: `supabase_client.py:608-621`, `routes.py:998-1016`
- **Endpoint**: `POST /api/v1/dashboard/clients/{id}/rotate-key`
- **Genera** nueva API key y la devuelve (solo visible una vez)

### ✅ 7. Graph Explorer API
- **Ubicación**: `routes.py:1054-1135`
- **Endpoint**: `GET /api/v1/dashboard/graph/{client_id}?limit=100`
- **Devuelve**: nodos y edges en formato vis.js/D3.js
- **Frontend**: Pendiente integrar librería de visualización

---

## Funcionalidades PENDIENTES

### 🟡 Prioridad Media

#### 8. Paginación Completa en Lista de Páginas
- **Estado**: Backend sí, frontend NO
- **Backend**: `list_pages()` tiene offset/limit ✅
- **Frontend**: NO implementa navegación
- **Solución**:
  - Añadir botones "Anterior/Siguiente" o scroll infinito

#### 9. Graph Explorer Visual (Frontend)
- **Estado**: API lista, falta frontend
- **Solución**: Integrar vis.js en index.html para visualizar el grafo

---

### 🟢 Prioridad Baja

#### 10. Export de Datos
- **Estado**: No existe
- **Solución**:
  - Endpoint `GET /admin/clients/{id}/export`
  - Formato: JSON o CSV

#### 11. Filtros Avanzados en Búsqueda
- **Estado**: Backend parcial, frontend NO
- **Solución**: Añadir controles de filtro en la interfaz

#### 12. Analytics/Estadísticas de Uso
- **Estado**: No existe
- **Solución**: Dashboard de estadísticas usando `rag_messages`

#### 13. Validación de Dominio en Crawl
- **Estado**: No existe
- **Solución**: Validar que el dominio coincida con el cliente

---

## Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `graph_rag/db/neo4j_client.py` | Cliente Neo4j - añadir PageRank |
| `graph_rag/db/supabase_client.py` | Cliente Supabase - añadir delete, etc |
| `graph_rag/services/rag_service.py` | Servicio RAG - conectar historial |
| `graph_rag/services/ingest_service.py` | Ingest - trigger recálculo scores |
| `graph_rag/api/routes.py` | Endpoints - añadir nuevos |
| `graph_rag/static/index.html` | Dashboard - Graph Explorer, paginación |

---

## Orden de Implementación Sugerido

1. **PageRank/HITS** - Sin esto el ranking no funciona correctamente
2. **Historial Chat** - Ya tienes toda la infraestructura
3. **Recálculo Post-Ingest** - Depende de #1
4. **Delete Client** - Funcionalidad básica necesaria
5. **Regenerar Embeddings** - Útil para actualizaciones de modelo
6. **API Key Rotation** - Seguridad
7. **Graph Explorer** - UX/visualización
8. Resto según necesidad

---

*Última actualización: 2025-11-30*
