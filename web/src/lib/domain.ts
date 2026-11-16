// Typed access to the JSON that scripts/gen-domain.mjs writes from domain/sentences.yaml.
import generated from "@/generated/domain.json";

export interface DomainSentence {
  id: string;
  text: string;
  role: string;
}

export interface DomainInfo {
  generated_from: string;
  domain: string;
  sign_language: string | null;
  text_language: string | null;
  counts: { sentences: number; train: number; unseen_test: number; vocabulary: number };
  sentences: DomainSentence[];
}

export const domainInfo: DomainInfo = generated;
