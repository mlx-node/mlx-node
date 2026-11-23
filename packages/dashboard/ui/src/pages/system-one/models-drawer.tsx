import type { ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { Alert } from '../../components/arc/alert/alert';
import { Button } from '../../components/arc/button/button';
import { Drawer, DrawerContent } from '../../components/arc/drawer/drawer';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import type { CatalogItem, CatalogResponse, CatalogUpdatesResponse, HostResponse } from '../../lib/types';
import type { AsyncState } from '../../lib/use-api';
import { PageSection } from '../../shell/page';
import { CatalogCardPlaceholder, CatalogEntryCard, CatalogGrid, useChosenBuilds } from '../models/catalog';
import type { ModelDownloads } from '../models/use-model-downloads';

export function isSystemOneCatalogItem(item: CatalogItem): boolean {
  if (item.category === 'systemone') return true;
  const hay = `${item.label} ${item.builds[0].hfRepo} ${item.builds[0].slug}`.toLowerCase();
  return hay.includes('clairic') || hay.includes('clef');
}

export interface SystemOneModelsDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  catalog: AsyncState<CatalogResponse>;
  host: AsyncState<HostResponse>;
  updates: AsyncState<CatalogUpdatesResponse>;
  downloads: ModelDownloads;
}

/** Drawer reusing the exact `Models` CatalogEntryCard + download progress flow for System One models. */
export function SystemOneModelsDrawer({
  open,
  onOpenChange,
  catalog,
  host,
  downloads,
}: SystemOneModelsDrawerProps): ReactElement {
  const items = (catalog.data?.items ?? []).filter((item) => !item.hidden && isSystemOneCatalogItem(item));
  const { chosen, choose } = useChosenBuilds();

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent
        side="right"
        title="System One models"
        description="Download Clairic Flash (9B), Clairic (27B), or CLEF checkpoints directly to this Mac."
      >
        <div className="grid min-w-0 gap-4">
          {catalog.error !== undefined ? (
            <Alert tone="danger" title="Could not load catalog">
              <span className="grid justify-items-start gap-2">
                <span>{catalog.error.message}</span>
                <Button variant="secondary" size="sm" onClick={catalog.reload}>
                  Try again
                </Button>
              </span>
            </Alert>
          ) : null}

          {catalog.data === undefined || host.loading ? (
            catalog.error !== undefined ? null : (
              <CatalogGrid>
                <CatalogCardPlaceholder />
                <CatalogCardPlaceholder />
              </CatalogGrid>
            )
          ) : items.length === 0 ? (
            <EmptyState
              title="No System One catalog entries"
              description="Check the Models page or install a Clairic checkpoint via the CLI."
              action={
                <Link className="link" to="/models">
                  Go to Models
                </Link>
              }
            />
          ) : (
            <CatalogGrid>
              {items.map((item) => (
                <CatalogEntryCard
                  key={item.label}
                  item={item}
                  downloads={downloads}
                  host={host.data}
                  chosen={chosen}
                  choose={choose}
                />
              ))}
            </CatalogGrid>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
}

/** Inline onboarding card displayed when 0 System One models are installed. */
export function NoSystemOneModelsCard({
  catalog,
  host,
  downloads,
}: {
  catalog: AsyncState<CatalogResponse>;
  host: AsyncState<HostResponse>;
  downloads: ModelDownloads;
}): ReactElement {
  const items = (catalog.data?.items ?? []).filter((item) => !item.hidden && isSystemOneCatalogItem(item));
  const { chosen, choose } = useChosenBuilds();

  return (
    <PageSection
      title="Install a System One model to get started"
      description="Download Clairic Flash (9B) for fast structured extraction or Clairic (27B) for highest accuracy and forecasting."
    >
      {(catalog.data === undefined && catalog.error === undefined) || host.loading ? (
        <CatalogGrid>
          <CatalogCardPlaceholder />
          <CatalogCardPlaceholder />
        </CatalogGrid>
      ) : items.length > 0 ? (
        <CatalogGrid>
          {items.map((item) => (
            <CatalogEntryCard
              key={item.label}
              item={item}
              downloads={downloads}
              host={host.data}
              chosen={chosen}
              choose={choose}
            />
          ))}
        </CatalogGrid>
      ) : (
        <EmptyState
          title="No System One models installed"
          description="Install a Clairic or CLEF checkpoint from Models or via `mlx download`."
          action={
            <Link className="link" to="/models">
              Open Models
            </Link>
          }
        />
      )}
    </PageSection>
  );
}
