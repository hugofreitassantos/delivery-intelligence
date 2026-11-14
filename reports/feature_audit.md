
# Auditoria de Variáveis — Delivery Intelligence

## Objetivo da previsão

Prever a probabilidade de atraso imediatamente após a criação
do pedido, antes do início da entrega.

## Critérios

- Manter: informação disponível no momento da previsão.
- Verificar: disponibilidade temporal ainda precisa ser confirmada.
- Excluir: informação posterior ao momento da previsão.

## Variáveis aprovadas inicialmente

- n_items
- n_distinct_products
- n_sellers
- total_price
- total_freight
- max_item_price
- total_weight_g
- max_volume_cm3
- avg_photos_qty
- avg_description_len
- purchase_year
- purchase_month
- purchase_day_of_week
- purchase_hour
- purchase_is_weekend
- purchase_week_of_year
- freight_ratio
- price_per_item
- freight_per_kg
- same_state
- is_sao_paulo_seller
- multi_seller_order
- weight_per_item_g
- primary_category
- customer_state
- seller_state

## Variáveis pendentes de confirmação

- total_payment_value
- max_installments
- n_payment_methods
- distance_km
- estimated_delivery_days
- payment_type

## Variáveis que não devem ser utilizadas

- actual_delivery_days
- days_late
- order_delivered_customer_date
- order_delivered_carrier_date
- order_status

## Observação

A classificação é preliminar. A aprovação definitiva depende
da confirmação de que cada informação estava disponível no
momento em que a previsão seria realizada.
