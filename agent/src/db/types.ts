export type SpaceObject = {
  id: string;
  location: string;
  description: string;
};

export type SpaceStatus = {
  detected_at: string;
  objects: SpaceObject[];
};

export interface SpaceStatusStore {
  getStatus(): Promise<SpaceStatus | null>;
}
